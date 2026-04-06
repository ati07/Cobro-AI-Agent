import json
import operator
from datetime import datetime, UTC
from typing import Any, Literal

from bson import ObjectId
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pymongo import MongoClient
from typing_extensions import Annotated, TypedDict
from app.core.database import get_db

db = get_db()


COBROS_VENTAS_SCHEMA = """
collectionReports (cobros):
- _id: ObjectId
- addedBy: ObjectId
- reservaBy: ObjectId
- recibidoBy: ObjectId
- cuentaId: ObjectId
- titleCuentaId: ObjectId
- projectId: ObjectId
- bankId: ObjectId
- clientId: ObjectId
- userId: ObjectId
- unitNameByInventoryId: ObjectId
- cobroNumber: number
- estadoDelCobro: string (default: Pendiente)
- reportDate: date
- collectionReportDate: date
- entryDate: date
- pagoCapital: string
- intereses: string (default: 0)
- totalCollection: string
- gastos: string (default: 0)
- fondosFaltantes: string (default: 0)
- fechaEntrada: date
- recibido: string (default: 0)
- typeOfPayment: string
- paymentDate: date
- observation: string
- isTipo: string (default: Venta)
- justificanteDePagoFilePath: string
- emailReminderCount: number (default: 0)
- lastEmailReminderAt: date
- isEmailedFromEntradas: boolean (default: false)
- isEmailedToClient: boolean (default: false)
- isEmailedToAnna: boolean (default: false)
- isComplete: boolean
- isDelete: boolean (default: false)
- isActive: boolean (default: true)
- createdAt: date
- updatedAt: date

ventas:
- _id: ObjectId
- addedBy: ObjectId
- projectId: ObjectId
- clientId: ObjectId
- userId: ObjectId
- statusId: ObjectId
- unitNameByInventoryId: ObjectId
- fechaDeReserva: date
- fechaDeVenta: date
- precioVenta: string
- financiamiento: string (default: No)
- intereses: string (default: 0)
- precioTotalVenta: string
- contractFilepath: string
- identificationFilepath: string
- observaciones: string
- comment: string
- payments: array
- otrosGastos: string (default: No)
- gastos: string (default: 0)
- descripcionDeOtrosGastos: string
- isEmailed: boolean (default: false)
- isPaid: boolean (default: false)
- isComplete: boolean
- isDelete: boolean (default: false)
- isActive: boolean (default: true)
- createdAt: date
- updatedAt: date

rentas:
- fechaDeRenta: date
- precioRenta, precioTotalRenta: string
- unitName: string
- clientId → clients, projectId → projects
- unitNameByInventoryId → inventories
- isDelete, isActive: boolean

reservas:
- fechaDeReserva, fechaDeVenta: date
- precioVenta, precioTotalVenta: string
- clientId → clients, projectId → projects
- unitNameByInventoryId → inventories
- isDelete, isActive: boolean
"""

CLIENT_SCHEMA = """
clients:
- name, email, phoneNumber, description
- tipoIdentificacion, identificacion, statusCliente
- comment, isComplete, isDelete, isActive
"""

PROJECT_SCHEMA = """
projects:
- name, empresa, ruc, dv, managerName, managerEmail
- isDelete, isActive
"""

INVENTORY_SCHEMA = """
inventories:
- unitName, code, met2, unitArea, priceUnit, priceList, rooms, view, comment
- projectId → projects, clientId → clients
- isComplete, isDelete, isActive
"""

BANK_SCHEMA = """
banks:
- name, isDelete, isActive
"""

PROVIDER_SCHEMA = """
providers:
- name, email, phoneNumber, serviceType, contactPerson
- snCode, snName
- projectId → projects
- isComplete, isDelete, isActive
"""

USER_SCHEMA = "users: _id, name, email, role, isActive, isDelete"

from typing import Optional, Dict, Any, List
from datetime import datetime


def serialize_mongo(obj):
    if isinstance(obj, list):
        return [serialize_mongo(i) for i in obj]
    if isinstance(obj, dict):
        return {k: serialize_mongo(v) for k, v in obj.items()}
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def to_double(field: str) -> dict:
    return {"$convert": {"input": field, "to": "double", "onError": 0, "onNull": 0}}


def parse_start_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str).replace(hour=0, minute=0, second=0, microsecond=0)


def parse_end_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str).replace(hour=23, minute=59, second=59, microsecond=999999)


def safe_object_id(value: str) -> ObjectId | str:
    try:
        return ObjectId(value) if len(value) == 24 else value
    except Exception:
        return value
    
def sanitize_pipeline(obj):
    """
    Recursively replace unsafe $toDouble with safe $convert.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "$toDouble":
                return {
                    "$convert": {
                        "input": sanitize_pipeline(v),
                        "to": "double",
                        "onError": 0,
                        "onNull": 0,
                    }
                }
            new_obj[k] = sanitize_pipeline(v)
        return new_obj

    elif isinstance(obj, list):
        return [sanitize_pipeline(i) for i in obj]

    return obj

COLLECTION_ALIASES = {
    "ventas": "ventas", "sales": "ventas",
    "CobrosProgramados": "ventas_with_payments_tool", "scheduled payments": "ventas_with_payments_tool",
    "cobros": "collectionreports", "cobro": "collectionreports",
    "collectionreports": "collectionreports", "payments": "collectionreports",
    "rentas": "rentas", "renta": "rentas", "rentals": "rentas",
    "reservas": "reservas", "reserva": "reservas", "reservations": "reservas",
}

def normalize_collection(name: str) -> str | None:
    return COLLECTION_ALIASES.get(name.lower())


# ─── Shared populate stages for ventas/cobros/rentas/reservas ─────────────────
POPULATE_NAMES_STAGES = [
    {"$lookup": {"from": "clients",     "localField": "clientId",              "foreignField": "_id", "as": "_client"}},
    {"$unwind": {"path": "$_client",    "preserveNullAndEmptyArrays": True}},
    {"$lookup": {"from": "projects",    "localField": "projectId",             "foreignField": "_id", "as": "_project"}},
    {"$unwind": {"path": "$_project",   "preserveNullAndEmptyArrays": True}},
    {"$lookup": {"from": "inventories", "localField": "unitNameByInventoryId", "foreignField": "_id", "as": "_inventory"}},
    {"$unwind": {"path": "$_inventory", "preserveNullAndEmptyArrays": True}},
    {"$lookup": {"from": "banks",       "localField": "bankId",                "foreignField": "_id", "as": "_bank"}},
    {"$unwind": {"path": "$_bank",      "preserveNullAndEmptyArrays": True}},
    {"$addFields": {
        "clientName":     {"$ifNull": ["$_client.name",         "Unknown Client"]},
        "clientEmail":    {"$ifNull": ["$_client.email",        ""]},
        "clientCompany":  {"$ifNull": ["$_client.company",      ""]},
        "clientPhone":    {"$ifNull": ["$_client.phoneNumber",  ""]},
        "projectName":    {"$ifNull": ["$_project.name",        "Unknown Project"]},
        "projectRuc":     {"$ifNull": ["$_project.ruc",         ""]},
        "projectDv":      {"$ifNull": ["$_project.dv",          ""]},
        "projectEmpresa": {"$ifNull": ["$_project.empresa",     ""]},
        "managerName":    {"$ifNull": ["$_project.managerName", ""]},
        "managerEmail":   {"$ifNull": ["$_project.managerEmail",""]},
        "unitName":       {"$ifNull": ["$_inventory.unitName",  "$unitName"]},
        "bankName":       {"$ifNull": ["$_bank.name",           ""]},
    }},
    {"$project": {
        "_id": 0, "_client": 0, "_project": 0, "_inventory": 0, "_bank": 0,
        "clientId": 0, "projectId": 0, "unitNameByInventoryId": 0,
        "userId": 0, "addedBy": 0, "statusId": 0,
        "bankId": 0, "cuentaId": 0, "titleCuentaId": 0,
        "reservaBy": 0, "recibidoBy": 0,
    }},
]

POPULATE_NAMES_ONLY = [
    {"$lookup": {"from": "clients", "localField": "clientId", "foreignField": "_id", "as": "_client"}},
    {"$unwind": {"path": "$_client", "preserveNullAndEmptyArrays": True}},

    {"$lookup": {"from": "projects", "localField": "projectId", "foreignField": "_id", "as": "_project"}},
    {"$unwind": {"path": "$_project", "preserveNullAndEmptyArrays": True}},

    {"$lookup": {"from": "inventories", "localField": "unitNameByInventoryId", "foreignField": "_id", "as": "_inventory"}},
    {"$unwind": {"path": "$_inventory", "preserveNullAndEmptyArrays": True}},

    {"$lookup": {"from": "banks", "localField": "bankId", "foreignField": "_id", "as": "_bank"}},
    {"$unwind": {"path": "$_bank", "preserveNullAndEmptyArrays": True}},

    {"$addFields": {
        "clientName": {"$ifNull": ["$_client.name", "Unknown Client"]},
        "projectName": {"$ifNull": ["$_project.name", "Unknown Project"]},
        "unitName": {"$ifNull": ["$_inventory.unitName", "$unitName"]},
        "bankName": {"$ifNull": ["$_bank.name", ""]},
    }},
]
# ══════════════════════════════════════════════════════════════════════════════
# ─── ENTITY TOOLS (clients, projects, inventories, banks, providers) ──────────
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_clients(
    filters: dict | None = None,
    limit: int = 10000,
) -> list:
    """
    Retrieve client records from the clients collection.
    Returns: name, email, phoneNumber, description, tipoIdentificacion,
             identificacion, statusCliente, comment, isComplete, isActive.
 
    Default limit is 10000 — effectively returns ALL clients unless filtered.
    Only reduce limit when the user explicitly asks for a specific number (e.g. "top 5").
 
    Optional filters examples:
      {"name": "JP Kramer"}         → find by name (use regex for partial)
      {"statusCliente": "Activo"}   → filter by status
      {"isActive": true}            → only active clients
    """
    base: dict = {"isDelete": False}
    if filters:
        for k, v in list(filters.items()):
            if (k.endswith("Id") or k == "_id") and isinstance(v, str):
                filters[k] = safe_object_id(v)
        base.update(filters)
 
    docs = list(db["clients"].find(base, {"_id": 0, "addedBy": 0}).limit(limit))
    # return serialize_mongo(docs)
    serialized = serialize_mongo(docs)
    return {"total": len(serialized), "clients": serialized}


@tool
def get_projects(
    filters: dict | None = None,
    limit: int = 10000,
) -> list:
    """
    Retrieve project records from the projects collection.
    Returns: name, empresa, ruc, dv, managerName, managerEmail, isActive.
    
    Default limit is 10000 — effectively returns ALL projects unless filtered.
    Only reduce limit when the user explicitly asks for a specific number.
 
    Optional filters examples:
      {"name": "Alamar"}            → find by project name
      {"isActive": true}            → only active projects
    """
    base: dict = {"isDelete": False, "isActive": True}
    if filters:
        base.update(filters)
 
    docs = list(db["projects"].find(base, {"_id": 0, "addedBy": 0, "logoFilePath": 0}).limit(limit))
    # return serialize_mongo(docs)
    serialized = serialize_mongo(docs)
    return {"total": len(serialized), "projects": serialized}


@tool
def get_inventory(
    filters: dict | None = None,
    project_id: str | None = None,
    client_id: str | None = None,
    limit: int = 10000,
) -> list:
    """
    Retrieve inventory (units) records with project and client names populated.
    Returns: unitName, code, met2, unitArea, priceUnit, priceList, rooms, view,
             comment, isComplete, isActive, projectName, clientName.
 
    Default limit is 10000 — effectively returns ALL units unless filtered.
    Only reduce limit when the user explicitly asks for a specific number.
 
    Optional filters examples:
      {"unitName": "7081"}          → find specific unit
      {"isComplete": false}         → available units
    Use project_id or client_id to filter by specific project or client ObjectId string.
    """
    base: dict = {"isDelete": False}
    if project_id:
        base["projectId"] = safe_object_id(project_id)
    if client_id:
        base["clientId"] = safe_object_id(client_id)
    if filters:
        for k, v in list(filters.items()):
            if (k.endswith("Id") or k == "_id") and isinstance(v, str):
                filters[k] = safe_object_id(v)
        base.update(filters)
 
    pipeline = [
        {"$match": base},
        {"$limit": limit},
        {"$lookup": {"from": "projects", "localField": "projectId", "foreignField": "_id", "as": "_project"}},
        {"$unwind": {"path": "$_project", "preserveNullAndEmptyArrays": True}},
        {"$lookup": {"from": "clients",  "localField": "clientId",  "foreignField": "_id", "as": "_client"}},
        {"$unwind": {"path": "$_client",  "preserveNullAndEmptyArrays": True}},
        {"$addFields": {
            "projectName": {"$ifNull": ["$_project.name", ""]},
            "clientName":  {"$ifNull": ["$_client.name",  ""]},
        }},
        {"$project": {
            "_id": 0, "addedBy": 0, "projectId": 0, "clientId": 0,
            "statusId": 0, "userId": 0, "typeId": 0,
            "_project": 0, "_client": 0,
        }},
    ]
    # return serialize_mongo(list(db["inventories"].aggregate(pipeline)))
    docs = serialize_mongo(list(db["inventories"].aggregate(pipeline)))
    return {"total": len(docs), "inventory": docs}

@tool
def get_banks(filters: dict | None = None, limit: int = 10000) -> list:
    """
    Retrieve bank records. Default limit 10000 — returns all banks.
    Returns: name, isActive.
    """
    base: dict = {"isDelete": False}
    if filters:
        base.update(filters)
    docs = list(db["banks"].find(base, {"_id": 0, "addedBy": 0}).limit(limit))
    # return serialize_mongo(docs)
    docs = serialize_mongo(docs)
    return {"total": len(docs), "banks": docs}

@tool
def get_providers(
    filters: dict | None = None,
    project_id: str | None = None,
    limit: int = 10000,
) -> list:
    """
    Retrieve provider records with project name populated.
    Returns: name, email, phoneNumber, serviceType, contactPerson,
             snCode, snName, projectName, isActive.
 
    Optional filters examples:
      {"serviceType": "Electricidad"} → filter by service type
    Use project_id to filter by specific project.
    """
    base: dict = {"isDelete": False}
    if project_id:
        base["projectId"] = safe_object_id(project_id)
    if filters:
        for k, v in list(filters.items()):
            if (k.endswith("Id") or k == "_id") and isinstance(v, str):
                filters[k] = safe_object_id(v)
        base.update(filters)
 
    pipeline = [
        {"$match": base},
        {"$limit": limit},
        {"$lookup": {"from": "projects", "localField": "projectId", "foreignField": "_id", "as": "_project"}},
        {"$unwind": {"path": "$_project", "preserveNullAndEmptyArrays": True}},
        {"$addFields": {"projectName": {"$ifNull": ["$_project.name", ""]}}},
        {"$project": {
            "_id": 0, "addedBy": 0, "projectId": 0, "_project": 0,
        }},
    ]
    # return serialize_mongo(list(db["providers"].aggregate(pipeline)))
    docs = serialize_mongo(list(db["providers"].aggregate(pipeline)))
    return {"total": len(docs), "providers": docs}
# ══════════════════════════════════════════════════════════════════════════════
# ─── TRANSACTION TOOLS (ventas, cobros, rentas, reservas) ────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@tool
def find_records(collection: str, filters: dict | None = None, limit: int = 10000):
    """
    Find records from ventas or collectionReports (cobros), rentas, or reservas.
    Automatically populates: clientName, clientEmail, clientCompany, clientPhone,
    projectName, projectRuc, projectDv, projectEmpresa, managerName, managerEmail,
    unitName, bankName. Never returns raw ObjectIds.

    collection: ventas | cobros | rentas | reservas
    """
    real_collection = normalize_collection(collection)
    if not real_collection:
        raise ValueError("Unknown collection. Use: ventas, cobros, rentas, reservas.")

    base_filter: dict = {"isDelete": False}
    if filters:
        for key, value in list(filters.items()):
            if (key.endswith("Id") or key == "_id") and isinstance(value, str):
                filters[key] = safe_object_id(value)
        base_filter.update(filters)

    pipeline = [{"$match": base_filter}, {"$limit": limit}] + POPULATE_NAMES_STAGES
    # return serialize_mongo(list(db[real_collection].aggregate(pipeline)))
    docs = serialize_mongo(list(db[real_collection].aggregate(pipeline)))
    return {"total": len(docs), "records": docs}


@tool
def aggregate_records(collection: str, pipeline: list[dict]) -> list:
    """Run custom aggregation pipelines for advanced reports."""
    real_collection = normalize_collection(collection)
    if not real_collection:
        raise ValueError("Unknown collection.")

    converted = []
    for stage in pipeline:
        if "$match" in stage:
            m = stage["$match"].copy()
            for key, value in m.items():
                if (key.endswith("Id") or key == "_id") and isinstance(value, str):
                    m[key] = safe_object_id(value)
            converted.append({"$match": m})
        else:
            converted.append(stage)
    # 🔥 sanitize entire pipeline (this is the fix)
    safe_pipeline = sanitize_pipeline(converted)
    # return serialize_mongo(list(db[real_collection].aggregate(converted)))
    return serialize_mongo(list(db[real_collection].aggregate(safe_pipeline)))



@tool
def ventas_report(start_date: str | None = None, end_date: str | None = None, summary: bool = True) -> dict:
    """
    Generate ventas report between two ISO dates (YYYY-MM-DD).
    summary=True  → totals: totalVentas count + totalMonto.
    summary=False → full detail rows with all names populated, no raw IDs.
    """
    # start = parse_start_date(start_date)
    # end   = parse_end_date(end_date)

    # pipeline: list = [{"$match": {"fechaDeVenta": {"$gte": start, "$lte": end}, "isDelete": False}}]
    # pipeline: list = [{"$match": {"isDelete": False}}]
    
    match_filter = {"isDelete": False}

    if start_date or end_date:
        date_filter = {}

        if start_date:
            date_filter["$gte"] = parse_start_date(start_date)

        if end_date:
            date_filter["$lte"] = parse_end_date(end_date)

        match_filter["fechaDeVenta"] = date_filter

    pipeline: list = [{"$match": match_filter}]
    

    if summary:
        pipeline.append({"$group": {
            "_id": None,
            "totalVentas": {"$sum": 1},
            "totalMonto":  {"$sum": to_double("$precioTotalVenta")},
        }})
    else:
        pipeline += POPULATE_NAMES_STAGES

    return serialize_mongo(list(db["ventas"].aggregate(pipeline)))


@tool
def cobros_report(start_date: str, end_date: str, summary: bool = True) -> dict:
    """
    Generate cobros report between two ISO dates (YYYY-MM-DD).
    Filters by collectionReportDate (the business date shown in the UI).
    summary=True  → totals: count, totalCapital, totalIntereses, totalCobrado.
    summary=False → full detail rows with all names populated, no raw IDs.
    """
    start = parse_start_date(start_date)
    end   = parse_end_date(end_date)

    pipeline: list = [{"$match": {"collectionReportDate": {"$gte": start, "$lte": end}, "isDelete": False}}]

    if summary:
        pipeline.append({"$group": {
            "_id": None,
            "totalCobros":    {"$sum": 1},
            "totalCapital":   {"$sum": to_double("$pagoCapital")},
            "totalIntereses": {"$sum": to_double("$intereses")},
            "totalGastos":    {"$sum": to_double("$gastos")},
            "totalCobrado":   {"$sum": to_double("$totalCollection")},
        }})
    else:
        pipeline += POPULATE_NAMES_STAGES

    return serialize_mongo(list(db["collectionreports"].aggregate(pipeline)))

def ventas_with_payments_report(
    client_id: Optional[str] = None,
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    summary: bool = True,
) -> Dict[str, Any]:
    """
    Replicates frontend logic:
    - Flattens ventas + payments
    - Sums payment.amount
    - Applies optional filters
    """

    match_filter: Dict[str, Any] = {"isDelete": False}

    if client_id:
        match_filter["clientId"] = ObjectId(client_id)

    if project_id:
        match_filter["projectId"] = ObjectId(project_id)

    if status:
        match_filter["status"] = status

    pipeline: List[Dict[str, Any]] = [
        {"$match": match_filter},
        {"$unwind": "$payments"},
    ]

    # ✅ Apply date filter ONLY if provided
    if start_date or end_date:
        date_filter = {}

        if start_date:
            date_filter["$gte"] = parse_start_date(start_date)

        if end_date:
            date_filter["$lte"] = parse_end_date(end_date)

        pipeline.append({
            "$match": {
                "payments.date": date_filter
            }
        })

    # ✅ Summary mode (like totalVentasData)
    if summary:
        pipeline.append({
            "$group": {
                "_id": None,
                "totalVentas": {
                    "$sum": {
                        "$toDouble": "$payments.amount"
                    }
                },
                "count": {"$sum": 1}
            }
        })

    # ✅ Detailed mode (like cobrosProgramadosData)
    else:
        pipeline += POPULATE_NAMES_ONLY
        # pipeline.append({
        #     "$project": {
        #         "_id": 1,
        #         "clientId": 1,
        #         "projectId": 1,
        #         "status": 1,
        #         "pagoCapital": {"$toDouble": "$payments.amount"},
        #         "totalCollection": {"$toDouble": "$payments.amount"},
        #         "collectionReportDate": "$payments.date",
        #         "hitoDePago": "$payments.hitoDePago",
        #     }
        # })
        pipeline.append({
                "$project": {
                    "_id": 1,

                    # ✅ Use names (NOT IDs)
                    "clientName": 1,
                    "projectName": 1,
                    "unitName": 1,
                    "bankName": 1,

                    "status": 1,

                    "pagoCapital": {"$toDouble": "$payments.amount"},
                    "totalCollection": {"$toDouble": "$payments.amount"},
                    "collectionReportDate": "$payments.date",
                    "hitoDePago": "$payments.hitoDePago",
                }
            })

    result = list(db["ventas"].aggregate(pipeline))
    return serialize_mongo(result)


@tool
def ventas_with_payments_tool(
    client_id: Optional[str] = None,
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    summary: bool = True,
) -> dict:
    """
    Returns ventas with payments aggregation.
    Automatically handles optional date filtering.
    """
    return ventas_with_payments_report(
        client_id,
        project_id,
        status,
        start_date,
        end_date,
        summary,
    )
    
@tool
def rentas_report(start_date: str, end_date: str, summary: bool = True) -> dict:
    """
    Generate rentas (rentals) report between two ISO dates (YYYY-MM-DD).
    Filters by fechaDeRenta.
    summary=True  → totals: count + totalMonto.
    summary=False → full detail rows with all names populated.
    """
    start = parse_start_date(start_date)
    end   = parse_end_date(end_date)

    pipeline: list = [{"$match": {"fechaDeRenta": {"$gte": start, "$lte": end}, "isDelete": False}}]

    if summary:
        pipeline.append({"$group": {
            "_id": None,
            "totalRentas": {"$sum": 1},
            "totalMonto":  {"$sum": to_double("$precioTotalRenta")},
        }})
    else:
        pipeline += POPULATE_NAMES_STAGES

    return serialize_mongo(list(db["rentas"].aggregate(pipeline)))


@tool
def reservas_report(start_date: str, end_date: str, summary: bool = True) -> dict:
    """
    Generate reservas (reservations) report between two ISO dates (YYYY-MM-DD).
    Filters by fechaDeReserva.
    summary=True  → totals: count + totalMonto.
    summary=False → full detail rows with all names populated.
    """
    start = parse_start_date(start_date)
    end   = parse_end_date(end_date)

    pipeline: list = [{"$match": {"fechaDeReserva": {"$gte": start, "$lte": end}, "isDelete": False}}]

    if summary:
        pipeline.append({"$group": {
            "_id": None,
            "totalReservas": {"$sum": 1},
            "totalMonto":    {"$sum": to_double("$precioTotalVenta")},
        }})
    else:
        pipeline += POPULATE_NAMES_STAGES

    return serialize_mongo(list(db["reservas"].aggregate(pipeline)))


@tool
def balance_report(
    project_id: str | None = None,
    client_id:  str | None = None,
    include_details: bool = True,
) -> dict:
    """
    Calculate the TOTAL outstanding balance (saldo por cobrar) for ALL ventas vs cobros.
    NO date filter — always reflects the full current state.
    A venta from 2019 that is still unpaid still contributes to today's balance.

    NEVER pass date parameters. When user mentions dates with 'balance', ignore them.

    Args:
        project_id:      Optional ObjectId string to filter by project.
        client_id:       Optional ObjectId string to filter by client.
        include_details: True = one row per unit. False = single summary totals row.

    For total saldo por cobrar → balance_report(include_details=False)
    Field 'totalBalance' in summary = total saldo por cobrar.
    """
    match_stage: dict = {"isDelete": False}
    if project_id:
        match_stage["projectId"] = safe_object_id(project_id)
    if client_id:
        match_stage["clientId"] = safe_object_id(client_id)

    pipeline = [
        {"$match": match_stage},
        {"$group": {
            "_id": {"clientId": "$clientId", "projectId": "$projectId", "unit": "$unitNameByInventoryId"},
            "id":                    {"$last": "$_id"},
            "clientId":              {"$last": "$clientId"},
            "projectId":             {"$last": "$projectId"},
            "unitNameByInventoryId": {"$last": "$unitNameByInventoryId"},
            "unitName":              {"$last": "$unitName"},
            "precioVenta":           {"$last": to_double("$precioVenta")},
            "interesesVenta":        {"$last": to_double("$intereses")},
            "totalVentas":           {"$last": to_double("$precioTotalVenta")},
            "isPaid":                {"$last": "$isPaid"},
            "ventasCount":           {"$sum": 1},
        }},
        {"$lookup": {
            "from": "collectionreports",
            "let": {"clientId": "$_id.clientId", "projectId": "$_id.projectId", "unit": "$_id.unit"},
            "pipeline": [
                {"$match": {"isDelete": False, "$expr": {"$and": [
                    {"$eq": ["$clientId",              "$$clientId"]},
                    {"$eq": ["$projectId",             "$$projectId"]},
                    {"$eq": ["$unitNameByInventoryId", "$$unit"]},
                ]}}},
                {"$group": {
                    "_id": None,
                    "totalRecibido":   {"$sum": to_double("$recibido")},
                    "totalIntereses":  {"$sum": to_double("$intereses")},
                    "collectionCount": {"$sum": 1},
                }},
            ],
            "as": "collections",
        }},
        {"$addFields": {"collections": {"$arrayElemAt": ["$collections", 0]}}},
        {"$addFields": {
            "totalRecibido":   {"$ifNull": ["$collections.totalRecibido",   0]},
            "totalIntereses":  {"$ifNull": ["$collections.totalIntereses",  0]},
            "collectionCount": {"$ifNull": ["$collections.collectionCount", 0]},
        }},
        {"$addFields": {"totalCapital": {"$subtract": ["$totalRecibido", "$totalIntereses"]}}},
        {"$addFields": {
            "balance": {"$cond": [
                "$isPaid", 0,
                {"$subtract": ["$totalVentas", {"$add": ["$totalCapital", "$totalIntereses"]}]},
            ]},
            "saldoCapital":   {"$subtract": ["$precioVenta",    "$totalCapital"]},
            "saldoIntereses": {"$subtract": ["$interesesVenta", "$totalIntereses"]},
        }},
        {"$lookup": {"from": "clients",     "localField": "clientId",              "foreignField": "_id", "as": "_client"}},
        {"$unwind": {"path": "$_client",    "preserveNullAndEmptyArrays": True}},
        {"$lookup": {"from": "projects",    "localField": "projectId",             "foreignField": "_id", "as": "_project"}},
        {"$unwind": {"path": "$_project",   "preserveNullAndEmptyArrays": True}},
        {"$lookup": {"from": "inventories", "localField": "unitNameByInventoryId",  "foreignField": "_id", "as": "_inventory"}},
        {"$unwind": {"path": "$_inventory", "preserveNullAndEmptyArrays": True}},
        {"$project": {
            "_id": 0, "id": 1,
            "clientName":     {"$ifNull": ["$_client.name",        "Unknown"]},
            "clientEmail":    {"$ifNull": ["$_client.email",       ""]},
            "projectName":    {"$ifNull": ["$_project.name",       "Unknown"]},
            "projectRuc":     {"$ifNull": ["$_project.ruc",        ""]},
            "projectDv":      {"$ifNull": ["$_project.dv",         ""]},
            "managerName":    {"$ifNull": ["$_project.managerName",""]},
            "unitName":       {"$ifNull": ["$_inventory.unitName",  "$unitName"]},
            "precioVenta": 1, "intereses": "$interesesVenta",
            "precioTotalVenta": "$totalVentas",
            "totalRecibido": 1, "totalCapital": 1, "totalIntereses": 1,
            "balance": 1, "ventasCount": 1, "collectionCount": 1,
            "saldoCapital": 1, "saldoIntereses": 1,
        }},
    ]

    if not include_details:
        pipeline.append({"$group": {
            "_id": None,
            "totalPrecioVenta":      {"$sum": "$precioVenta"},
            "totalInteresesVenta":   {"$sum": "$intereses"},
            "totalPrecioTotalVenta": {"$sum": "$precioTotalVenta"},
            "totalRecibido":         {"$sum": "$totalRecibido"},
            "totalCapital":          {"$sum": "$totalCapital"},
            "totalIntereses":        {"$sum": "$totalIntereses"},
            "totalBalance":          {"$sum": "$balance"},
            "totalSaldoCapital":     {"$sum": "$saldoCapital"},
            "totalSaldoIntereses":   {"$sum": "$saldoIntereses"},
        }})

    return serialize_mongo(list(db["ventas"].aggregate(pipeline)))


@tool
def resolve_entity_by_name(entity_type: str, name: str) -> dict | None:
    """
    Resolve a client, project, bank, provider, or user by name to get their _id and details.
    Always call this first when user refers to any entity by name.

    entity_type: client | clients | project | projects | bank | banks |
                 provider | providers | user | users | inventory | inventories
    name: case-insensitive partial match
    """
    collection_map = {
        "client": "clients",       "clients": "clients",
        "project": "projects",     "projects": "projects",
        "inventory": "inventories","inventories": "inventories",
        "bank": "banks",           "banks": "banks",
        "provider": "providers",   "providers": "providers",
        "user": "users",           "users": "users",
    }
    col = collection_map.get(entity_type.lower())
    if not col:
        raise ValueError(f"Unknown entity type: '{entity_type}'. Use: client, project, bank, provider, user, inventory.")

    doc = db[col].find_one({"name": {"$regex": name, "$options": "i"}, "isDelete": False})
    return serialize_mongo(doc) if doc else None


# ─── Model ────────────────────────────────────────────────────────────────────
tools = [
    # Entity tools
    get_clients, get_projects, get_inventory, get_banks, get_providers,
    # Transaction tools
    find_records, aggregate_records,
    ventas_report, ventas_with_payments_tool, cobros_report, rentas_report, reservas_report,
    balance_report,
    # Utility
    resolve_entity_by_name,
]
tools_by_name = {t.name: t for t in tools}