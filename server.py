from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi import Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from fastapi.concurrency import run_in_threadpool
import os
import re
import json
import logging
import unicodedata
import uuid
from pathlib import Path
from typing import List, Optional, Literal
from datetime import datetime, timezone, timedelta

import jwt
import requests
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives import hashes
from urllib.parse import parse_qsl
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.serialization import load_der_public_key
import base64
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr, field_validator

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("listamarket")

# ---------------------------------------------------------------- config
mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

JWT_SECRET = os.environ.get("JWT_SECRET", "listamarket-dev-secret-change-in-prod-32bytes")
JWT_ALGORITHM = "HS256"
TOKEN_MINUTES = 60 * 24 * 60  # 60 dias: no mercado o sinal é ruim, evita relogar

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OCR_MODEL = os.environ.get("OCR_MODEL", "claude-haiku-5-5")
OCR_FREE_MONTHLY = int(os.environ.get("OCR_FREE_MONTHLY", "10"))
OCR_PREMIUM_MONTHLY = int(os.environ.get("OCR_PREMIUM_MONTHLY", "300"))
OCR_MAX_B64_CHARS = 8_000_000  # ~6 MB de imagem

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

app = FastAPI(title="Lista Market API")
api = APIRouter(prefix="/api")


# ---------------------------------------------------------------- helpers
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def name_key(name: str) -> str:
    """Chave de comparação: minúsculo, sem acento, sem espaços repetidos.
    'Leite  Integral' e 'leite integral' viram o mesmo produto."""
    n = unicodedata.normalize("NFKD", name.strip().lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", n)


def make_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    if not creds:
        raise HTTPException(status_code=401, detail="Não autenticado")
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Sessão inválida")
    user = await db.users.find_one({"id": payload.get("sub")}, {"_id": 0, "password_hash": 0})
    if not user:
        raise HTTPException(status_code=401, detail="Usuário não encontrado")
    return user


def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def ocr_quota(user: dict) -> dict:
    used = user.get("ocr_count", 0) if user.get("ocr_month") == current_month() else 0
    limit = OCR_PREMIUM_MONTHLY if user.get("plan") == "premium" else OCR_FREE_MONTHLY
    return {"used": used, "limit": limit, "remaining": max(limit - used, 0)}


def validate_password(password: str) -> str:
    if len(password) < 6:
        raise ValueError("A senha precisa ter no mínimo 6 caracteres")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        raise ValueError("A senha precisa ter letras e números")
    return password


# ---------------------------------------------------------------- models
class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=6, max_length=72)

    @field_validator("password")
    @classmethod
    def _pwd(cls, v: str) -> str:
        return validate_password(v)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class ListIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    store: Optional[str] = Field(default=None, max_length=120)


class ListUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    store: Optional[str] = Field(default=None, max_length=120)


class ItemIn(BaseModel):
    # ou `name` (item digitado) ou `product_id` (produto cadastrado)
    name: Optional[str] = Field(default=None, max_length=200)
    product_id: Optional[str] = None
    quantity: Optional[float] = Field(default=None, gt=0, le=100000)
    unit: str = Field(default="un", max_length=10)
    unit_price: Optional[float] = Field(default=None, ge=0, le=1_000_000)


class ProductIn(BaseModel):
    """Produto cadastrado pelo usuário. No PUT todos os campos são
    reenviados: o que vier vazio (null) limpa o campo."""
    name: str = Field(min_length=1, max_length=150)
    brand: Optional[str] = Field(default=None, max_length=80)
    quantity: Optional[float] = Field(default=None, gt=0, le=100000)
    weight: Optional[float] = Field(default=None, gt=0, le=1_000_000)
    weight_unit: Optional[Literal["g", "kg", "ml", "l"]] = None
    price: Optional[float] = Field(default=None, ge=0, le=1_000_000)


class ItemUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=150)
    quantity: Optional[float] = Field(default=None, gt=0, le=100000)
    unit: Optional[str] = Field(default=None, max_length=10)
    unit_price: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    clear_price: bool = False  # true = apagar o preço do item
    checked: Optional[bool] = None


class OcrIn(BaseModel):
    image_base64: str = Field(min_length=100)
    media_type: Literal["image/jpeg", "image/png", "image/webp"] = "image/jpeg"


# ---------------------------------------------------------------- produtos & histórico
def fmt_num(n: float) -> str:
    return f"{n:g}".replace(".", ",")


def product_key(name: str, brand: Optional[str] = None,
                weight: Optional[float] = None, weight_unit: Optional[str] = None) -> str:
    """Identidade do produto: nome + marca + peso. 'Arroz Tio João 5 kg' e
    'Arroz Tio João 1 kg' são produtos diferentes (preços não se comparam)."""
    parts = [name_key(name)]
    if brand and brand.strip():
        parts.append(name_key(brand))
    if weight and weight_unit:
        parts.append(f"{weight:g}{weight_unit}")
    return "|".join(parts)


def product_label(p: dict) -> str:
    parts = [p["name"]]
    if p.get("brand"):
        parts.append(p["brand"])
    if p.get("weight") and p.get("weight_unit"):
        parts.append(f"{fmt_num(p['weight'])} {p['weight_unit']}")
    return " ".join(parts)


def product_view(p: dict) -> dict:
    d = {k: v for k, v in p.items() if k not in ("_id", "user_id", "name_key")}
    d["label"] = product_label(p)
    return d


async def get_or_create_product(user_id: str, name: str) -> dict:
    key = product_key(name)
    prod = await db.products.find_one({"user_id": user_id, "name_key": key}, {"_id": 0})
    if prod:
        return prod
    now = now_iso()
    prod = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "name": name.strip(),
        "name_key": key,
        "brand": None, "quantity": None, "weight": None, "weight_unit": None, "price": None,
        "last_price": None, "last_store": None, "last_date": None,
        "min_price": None, "min_store": None, "min_date": None,
        "created_at": now, "updated_at": now,
    }
    try:
        await db.products.insert_one(dict(prod))
    except DuplicateKeyError:
        prod = await db.products.find_one({"user_id": user_id, "name_key": key}, {"_id": 0})
    return prod


async def refresh_product_stats(product_id: str) -> None:
    """Recalcula último preço pago e menor preço pago a partir do histórico."""
    last = await db.price_history.find_one(
        {"product_id": product_id}, {"_id": 0}, sort=[("date", -1)]
    )
    low = await db.price_history.find_one(
        {"product_id": product_id}, {"_id": 0}, sort=[("price", 1), ("date", -1)]
    )
    await db.products.update_one(
        {"id": product_id},
        {"$set": {
            "last_price": last["price"] if last else None,
            "last_store": last.get("store") if last else None,
            "last_date": last["date"] if last else None,
            "min_price": low["price"] if low else None,
            "min_store": low.get("store") if low else None,
            "min_date": low["date"] if low else None,
        }},
    )


async def record_price(user_id: str, item: dict, store: Optional[str]) -> None:
    """Grava (ou atualiza) o preço pago de um item marcado como comprado.
    Uma entrada por item: marcar/desmarcar/editar não duplica o histórico."""
    fields = {
        "user_id": user_id,
        "product_id": item["product_id"],
        "product_name": item["name"],
        "price": float(item["unit_price"]),
        "quantity": item["quantity"],
        "unit": item.get("unit", "un"),
        "store": store or None,
        "list_id": item["list_id"],
    }
    existing = await db.price_history.find_one({"item_id": item["id"]}, {"_id": 0})
    if existing:
        await db.price_history.update_one({"item_id": item["id"]}, {"$set": fields})
    else:
        await db.price_history.insert_one({
            "id": str(uuid.uuid4()), "item_id": item["id"], "date": now_iso(), **fields,
        })
    await refresh_product_stats(item["product_id"])


async def remove_price(item_id: str) -> None:
    entry = await db.price_history.find_one({"item_id": item_id}, {"_id": 0})
    if entry:
        await db.price_history.delete_one({"item_id": item_id})
        await refresh_product_stats(entry["product_id"])


def product_hint(prod: Optional[dict]) -> Optional[dict]:
    if not prod or prod.get("last_price") is None:
        return None
    return {
        "last_price": prod["last_price"], "last_store": prod.get("last_store"),
        "last_date": prod.get("last_date"),
        "min_price": prod.get("min_price"), "min_store": prod.get("min_store"),
        "min_date": prod.get("min_date"),
    }


def item_view(item: dict, prod: Optional[dict]) -> dict:
    price = item.get("unit_price")
    return {
        "id": item["id"], "list_id": item["list_id"], "product_id": item["product_id"],
        "name": item["name"], "quantity": item["quantity"], "unit": item.get("unit", "un"),
        "unit_price": price, "checked": item.get("checked", False),
        "subtotal": round(item["quantity"] * price, 2) if price is not None else None,
        "hint": product_hint(prod),
        "updated_at": item.get("updated_at"),
    }


def list_summary(lst: dict, items: List[dict]) -> dict:
    total = sum(i["quantity"] * i["unit_price"] for i in items if i.get("unit_price") is not None)
    total_checked = sum(
        i["quantity"] * i["unit_price"]
        for i in items if i.get("checked") and i.get("unit_price") is not None
    )
    return {
        "id": lst["id"], "name": lst["name"], "store": lst.get("store"),
        "status": lst.get("status", "open"),
        "created_at": lst["created_at"], "updated_at": lst["updated_at"],
        "finished_at": lst.get("finished_at"),
        "item_count": len(items),
        "checked_count": sum(1 for i in items if i.get("checked")),
        "missing_price": sum(1 for i in items if i.get("unit_price") is None),
        "total": round(total, 2),
        "total_checked": round(total_checked, 2),
    }


async def get_list_or_404(user: dict, list_id: str) -> dict:
    lst = await db.lists.find_one({"id": list_id, "user_id": user["id"]}, {"_id": 0})
    if not lst:
        raise HTTPException(status_code=404, detail="Lista não encontrada")
    return lst


async def get_item_or_404(user: dict, item_id: str) -> dict:
    item = await db.items.find_one({"id": item_id, "user_id": user["id"]}, {"_id": 0})
    if not item:
        raise HTTPException(status_code=404, detail="Item não encontrado")
    return item


async def touch_list(list_id: str) -> None:
    await db.lists.update_one({"id": list_id}, {"$set": {"updated_at": now_iso()}})


# ---------------------------------------------------------------- auth
def user_out(user: dict) -> dict:
    return {
        "id": user["id"], "email": user["email"], "name": user["name"],
        "plan": user.get("plan", "gratis"),
        "ocr": ocr_quota(user),
    }


@api.post("/auth/register", status_code=201)
async def register(body: RegisterIn):
    email = body.email.lower()
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=409, detail="Email já cadastrado")
    doc = {
        "id": str(uuid.uuid4()), "email": email, "name": body.name.strip(),
        "password_hash": pwd_ctx.hash(body.password),
        "plan": "gratis", "created_at": now_iso(),
    }
    await db.users.insert_one(dict(doc))
    return {"access_token": make_token(doc["id"]), "user": user_out(doc)}


@api.post("/auth/login")
async def login(body: LoginIn):
    user = await db.users.find_one({"email": body.email.lower()}, {"_id": 0})
    if not user or not pwd_ctx.verify(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email ou senha incorretos")
    return {"access_token": make_token(user["id"]), "user": user_out(user)}


@api.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return user_out(user)


@api.delete("/auth/me", status_code=204)
async def delete_account(user: dict = Depends(get_current_user)):
    """Exclui a conta e TODOS os dados do usuário (exigência de lojas/LGPD)."""
    uid = user["id"]
    await db.price_history.delete_many({"user_id": uid})
    await db.items.delete_many({"user_id": uid})
    await db.lists.delete_many({"user_id": uid})
    await db.ad_rewards.delete_many({"user_id": uid})
    await db.products.delete_many({"user_id": uid})
    await db.users.delete_one({"id": uid})
    return None


# ---------------------------------------------------------------- listas
@api.get("/lists")
async def get_lists(status: Optional[Literal["open", "done"]] = None,
                    user: dict = Depends(get_current_user)):
    query: dict = {"user_id": user["id"]}
    if status == "open":
        query["status"] = {"$ne": "done"}
    elif status == "done":
        query["status"] = "done"
    lists = await db.lists.find(query, {"_id": 0}).sort("updated_at", -1).to_list(500)
    by_list: dict = {l["id"]: [] for l in lists}
    if by_list:
        items = await db.items.find({"list_id": {"$in": list(by_list)}}, {"_id": 0}).to_list(50000)
        for it in items:
            by_list[it["list_id"]].append(it)
    return [list_summary(l, by_list[l["id"]]) for l in lists]


# Recompensas de anúncios: o servidor só reconhece registros com verified=True.
# Nenhuma rota pública concede recompensas sem validação da plataforma de anúncios.
# A integração com o provedor e a confirmação segura ainda serão implementadas.
# Limites do plano gratuito (validados no servidor).
FREE_MAX_LISTS = 1
FREE_MAX_ITEMS = 10
REWARD_ITEMS_PER_AD = 2
REWARD_DURATION_HOURS = 24
ADMOB_REWARDED_AD_UNIT_ID = "ca-app-pub-4150349574154625/6264835990"
ADMOB_SSV_KEYS_URL = "https://www.gstatic.com/admob/reward/verifier-keys.json"

def fetch_admob_public_keys():
    response = requests.get(ADMOB_SSV_KEYS_URL, timeout=10)
    response.raise_for_status()
    return response.json()["keys"]

def load_admob_public_key(key_data):
    key_bytes = base64.b64decode(key_data["base64"])
    return load_der_public_key(key_bytes)

def find_admob_public_key(keys, key_id):
    for key_data in keys:
        if str(key_data.get("keyId")) == str(key_id):
            return load_admob_public_key(key_data)
    raise ValueError("Chave pública do AdMob não encontrada")
def decode_admob_signature(signature):
    padding = "=" * (-len(signature) % 4)
    signature_bytes = base64.urlsafe_b64decode(signature + padding)

    if not signature_bytes:
        raise ValueError("Assinatura do AdMob inválida")

    return signature_bytes

def verify_admob_signature(signed_content, signature, public_key):
    decoded_signature = decode_admob_signature(signature)
    public_key.verify(
        decoded_signature,
        signed_content.encode("utf-8"),
        ec.ECDSA(hashes.SHA256()),
    )

def is_premium(user: dict) -> bool:
    return user.get("plan") == "premium"


async def check_list_limit(user: dict) -> None:
    if is_premium(user):
        return
    count = await db.lists.count_documents({"user_id": user["id"]})
    if count >= FREE_MAX_LISTS:
        raise HTTPException(
            status_code=403,
            detail="O plano Grátis permite apenas 1 lista. Exclua a lista atual ou assine o Premium para criar listas ilimitadas.",
        )


async def item_allowance(user: dict, list_id: str) -> dict:
    """Apenas recompensas verificadas e ainda válidas ampliam o limite."""
    if is_premium(user):
        return {"plan": "premium", "base_limit": None, "bonus_items": None, "item_limit": None}
    active = await db.ad_rewards.count_documents({
        "user_id": user["id"], "list_id": list_id,
        "expires_at": {"$gt": datetime.now(timezone.utc)},
        "verified": True,
    })
    bonus = active * REWARD_ITEMS_PER_AD
    return {"plan": "gratis", "base_limit": FREE_MAX_ITEMS,
            "bonus_items": bonus, "item_limit": FREE_MAX_ITEMS + bonus}


@api.get("/lists/{list_id}/item-allowance")
async def get_item_allowance(list_id: str, user: dict = Depends(get_current_user)):
    await get_list_or_404(user, list_id)
    allowance = await item_allowance(user, list_id)
    allowance["item_count"] = await db.items.count_documents({
        "user_id": user["id"], "list_id": list_id})
    allowance["reward_items_per_ad"] = REWARD_ITEMS_PER_AD
    allowance["reward_duration_hours"] = REWARD_DURATION_HOURS
    return allowance


async def check_item_limit(user: dict, list_id: str) -> None:
    if is_premium(user):
        return
    count = await db.items.count_documents({"user_id": user["id"], "list_id": list_id})
    allowance = await item_allowance(user, list_id)
    if count >= allowance["item_limit"]:
        raise HTTPException(
            status_code=403,
            detail="Limite de produtos atingido. Assista a um anúncio recompensado para liberar mais 2 produtos por 24 horas, exclua um item ou assine o Premium.",
        )


@api.post("/lists", status_code=201)
async def create_list(body: ListIn, user: dict = Depends(get_current_user)):
    await check_list_limit(user)
    now = now_iso()
    doc = {
        "id": str(uuid.uuid4()), "user_id": user["id"], "name": body.name.strip(),
        "store": (body.store or "").strip() or None, "status": "open",
        "created_at": now, "updated_at": now, "finished_at": None,
    }
    await db.lists.insert_one(dict(doc))
    return list_summary(doc, [])


@api.get("/lists/{list_id}")
async def get_list(list_id: str, user: dict = Depends(get_current_user)):
    lst = await get_list_or_404(user, list_id)
    items = await db.items.find({"list_id": list_id}, {"_id": 0}).sort("created_at", 1).to_list(5000)
    pids = list({i["product_id"] for i in items})
    prods = {}
    if pids:
        for p in await db.products.find({"id": {"$in": pids}}, {"_id": 0}).to_list(len(pids)):
            prods[p["id"]] = p
    return {**list_summary(lst, items), "items": [item_view(i, prods.get(i["product_id"])) for i in items]}


@api.put("/lists/{list_id}")
async def update_list(list_id: str, body: ListUpdate, user: dict = Depends(get_current_user)):
    lst = await get_list_or_404(user, list_id)
    upd: dict = {"updated_at": now_iso()}
    if body.name is not None:
        upd["name"] = body.name.strip()
    if body.store is not None:
        upd["store"] = body.store.strip() or None
    await db.lists.update_one({"id": list_id}, {"$set": upd})
    if "store" in upd:
        # o mercado informado vale para os preços já registrados desta lista
        await db.price_history.update_many({"list_id": list_id}, {"$set": {"store": upd["store"]}})
        pids = await db.price_history.distinct("product_id", {"list_id": list_id})
        for pid in pids:
            await refresh_product_stats(pid)
    lst.update(upd)
    items = await db.items.find({"list_id": list_id}, {"_id": 0}).to_list(5000)
    return list_summary(lst, items)


@api.delete("/lists/{list_id}", status_code=204)
async def delete_list(list_id: str, user: dict = Depends(get_current_user)):
    """Apaga a lista e os itens. O histórico de preços é mantido (é o que
    você pagou de verdade); para apagar um preço, use DELETE /history/{id}."""
    await get_list_or_404(user, list_id)
    await db.items.delete_many({"list_id": list_id})
    await db.lists.delete_one({"id": list_id})
    await db.ad_rewards.delete_many({"user_id": user["id"], "list_id": list_id})
    return None


@api.post("/lists/{list_id}/finish")
async def finish_list(list_id: str, user: dict = Depends(get_current_user)):
    lst = await get_list_or_404(user, list_id)
    items = await db.items.find({"list_id": list_id}, {"_id": 0}).to_list(5000)
    for it in items:
        if it.get("checked") and it.get("unit_price") is not None:
            await record_price(user["id"], it, lst.get("store"))
    now = now_iso()
    await db.lists.update_one({"id": list_id}, {"$set": {"status": "done", "finished_at": now, "updated_at": now}})
    lst.update({"status": "done", "finished_at": now, "updated_at": now})
    return list_summary(lst, items)


@api.post("/lists/{list_id}/reopen")
async def reopen_list(list_id: str, user: dict = Depends(get_current_user)):
    lst = await get_list_or_404(user, list_id)
    now = now_iso()
    await db.lists.update_one({"id": list_id}, {"$set": {"status": "open", "finished_at": None, "updated_at": now}})
    lst.update({"status": "open", "finished_at": None, "updated_at": now})
    items = await db.items.find({"list_id": list_id}, {"_id": 0}).to_list(5000)
    return list_summary(lst, items)


@api.post("/lists/{list_id}/duplicate", status_code=201)
async def duplicate_list(list_id: str, user: dict = Depends(get_current_user)):
    """Nova lista com os mesmos itens, já com o ÚLTIMO PREÇO pago em cada um.
    É o atalho da "próxima ida ao mercado"."""
    src = await get_list_or_404(user, list_id)
    await check_list_limit(user)
    src_items = await db.items.find({"list_id": list_id}, {"_id": 0}).sort("created_at", 1).to_list(5000)
    pids = list({i["product_id"] for i in src_items})
    prods = {}
    if pids:
        for p in await db.products.find({"id": {"$in": pids}}, {"_id": 0}).to_list(len(pids)):
            prods[p["id"]] = p
    now = now_iso()
    new_list = {
        "id": str(uuid.uuid4()), "user_id": user["id"], "name": f"{src['name']} (cópia)"[:120],
        "store": src.get("store"), "status": "open",
        "created_at": now, "updated_at": now, "finished_at": None,
    }
    await db.lists.insert_one(dict(new_list))
    new_items = []
    for it in src_items:
        prod = prods.get(it["product_id"])
        price = prod["last_price"] if prod and prod.get("last_price") is not None else it.get("unit_price")
        new_items.append({
            "id": str(uuid.uuid4()), "list_id": new_list["id"], "user_id": user["id"],
            "product_id": it["product_id"], "name": it["name"],
            "quantity": it["quantity"], "unit": it.get("unit", "un"),
            "unit_price": price, "checked": False, "created_at": now, "updated_at": now,
        })
    if new_items:
        await db.items.insert_many([dict(i) for i in new_items])
    return list_summary(new_list, new_items)


# ---------------------------------------------------------------- itens
@api.post("/lists/{list_id}/items", status_code=201)
async def add_item(list_id: str, body: ItemIn, user: dict = Depends(get_current_user)):
    await get_list_or_404(user, list_id)
    await check_item_limit(user, list_id)
    if body.product_id:
        prod = await db.products.find_one({"id": body.product_id, "user_id": user["id"]}, {"_id": 0})
        if not prod:
            raise HTTPException(status_code=404, detail="Produto não encontrado")
        name = (body.name or "").strip() or product_label(prod)
        quantity = body.quantity if body.quantity is not None else (prod.get("quantity") or 1)
        if body.unit_price is not None:
            price = body.unit_price
        else:  # preço cadastrado no produto; senão o último que você pagou
            price = prod["price"] if prod.get("price") is not None else prod.get("last_price")
    else:
        if not body.name or not body.name.strip():
            raise HTTPException(status_code=400, detail="Informe o nome do item")
        prod = await get_or_create_product(user["id"], body.name)
        name = body.name.strip()
        quantity = body.quantity if body.quantity is not None else 1
        price = body.unit_price
    now = now_iso()
    item = {
        "id": str(uuid.uuid4()), "list_id": list_id, "user_id": user["id"],
        "product_id": prod["id"], "name": name,
        "quantity": quantity, "unit": body.unit.strip() or "un",
        "unit_price": price, "checked": False,
        "created_at": now, "updated_at": now,
    }
    await db.items.insert_one(dict(item))
    await touch_list(list_id)
    return item_view(item, prod)


@api.put("/items/{item_id}")
async def update_item(item_id: str, body: ItemUpdate, user: dict = Depends(get_current_user)):
    item = await get_item_or_404(user, item_id)
    lst = await get_list_or_404(user, item["list_id"])
    upd: dict = {}
    if not await db.products.find_one({"id": item["product_id"]}, {"_id": 0, "id": 1}):
        # o produto foi apagado do catálogo: religa o item a um produto com o mesmo nome
        relinked = await get_or_create_product(user["id"], item["name"])
        item["product_id"] = relinked["id"]
        upd["product_id"] = relinked["id"]
    if body.name is not None and name_key(body.name) != name_key(item["name"]):
        prod = await get_or_create_product(user["id"], body.name)
        upd["product_id"] = prod["id"]
        upd["name"] = body.name.strip()
    elif body.name is not None:
        upd["name"] = body.name.strip()
    if body.quantity is not None:
        upd["quantity"] = body.quantity
    if body.unit is not None:
        upd["unit"] = body.unit.strip() or "un"
    if body.clear_price:
        upd["unit_price"] = None
    elif body.unit_price is not None:
        upd["unit_price"] = body.unit_price
    if body.checked is not None:
        upd["checked"] = body.checked
    upd["updated_at"] = now_iso()

    old_product_id = item["product_id"]
    await db.items.update_one({"id": item_id}, {"$set": upd})
    item.update(upd)

    if item.get("checked") and item.get("unit_price") is not None:
        await record_price(user["id"], item, lst.get("store"))
    else:
        await remove_price(item_id)
    if item["product_id"] != old_product_id:
        await refresh_product_stats(old_product_id)

    await touch_list(item["list_id"])
    prod = await db.products.find_one({"id": item["product_id"]}, {"_id": 0})
    return item_view(item, prod)


@api.delete("/items/{item_id}", status_code=204)
async def delete_item(item_id: str, user: dict = Depends(get_current_user)):
    item = await get_item_or_404(user, item_id)
    await remove_price(item_id)
    await db.items.delete_one({"id": item_id})
    await touch_list(item["list_id"])
    return None


# ---------------------------------------------------------------- produtos & histórico
@api.get("/products")
async def search_products(q: Optional[str] = None, limit: int = 20,
                          sort: Literal["recent", "name"] = "recent",
                          user: dict = Depends(get_current_user)):
    """Lista/busca os produtos. sort=recent (padrão, bom para autocomplete)
    traz os comprados mais recentemente; sort=name é ordem alfabética."""
    limit = max(1, min(limit, 200))
    query: dict = {"user_id": user["id"]}
    if q and q.strip():
        query["name_key"] = {"$regex": re.escape(name_key(q))}
    order = [("name_key", 1)] if sort == "name" else [("last_date", -1), ("created_at", -1)]
    docs = await db.products.find(query, {"_id": 0}).sort(order).to_list(limit)
    return [product_view(d) for d in docs]


def product_fields(body: ProductIn) -> dict:
    return {
        "name": body.name.strip(),
        "brand": (body.brand or "").strip() or None,
        "quantity": body.quantity,
        "weight": body.weight,
        "weight_unit": body.weight_unit if body.weight else None,
        "price": body.price,
    }


@api.post("/products", status_code=201)
async def create_product(body: ProductIn, user: dict = Depends(get_current_user)):
    f = product_fields(body)
    key = product_key(f["name"], f["brand"], f["weight"], f["weight_unit"])
    if await db.products.find_one({"user_id": user["id"], "name_key": key}):
        raise HTTPException(status_code=409, detail="Você já tem esse produto cadastrado")
    now = now_iso()
    prod = {
        "id": str(uuid.uuid4()), "user_id": user["id"], "name_key": key, **f,
        "last_price": None, "last_store": None, "last_date": None,
        "min_price": None, "min_store": None, "min_date": None,
        "created_at": now, "updated_at": now,
    }
    await db.products.insert_one(dict(prod))
    return product_view(prod)


@api.put("/products/{product_id}")
async def update_product(product_id: str, body: ProductIn, user: dict = Depends(get_current_user)):
    prod = await db.products.find_one({"id": product_id, "user_id": user["id"]}, {"_id": 0})
    if not prod:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    f = product_fields(body)
    key = product_key(f["name"], f["brand"], f["weight"], f["weight_unit"])
    clash = await db.products.find_one({"user_id": user["id"], "name_key": key, "id": {"$ne": product_id}})
    if clash:
        raise HTTPException(status_code=409, detail="Você já tem outro produto igual a esse")
    upd = {**f, "name_key": key, "updated_at": now_iso()}
    await db.products.update_one({"id": product_id, "user_id": user["id"]}, {"$set": upd})

    # Propaga mudanças de preço apenas para listas em andamento.
    # Listas concluídas e o histórico de preços pagos não são modificados.
    if f["price"] != prod.get("price"):
        open_lists = await db.lists.find(
            {"user_id": user["id"], "status": {"$ne": "done"}},
            {"_id": 0, "id": 1},
        ).to_list(50000)
        open_ids = [lst["id"] for lst in open_lists]
        if open_ids:
            changed_at = now_iso()
            await db.items.update_many(
                {"user_id": user["id"], "product_id": product_id,
                 "list_id": {"$in": open_ids}},
                {"$set": {"unit_price": f["price"], "updated_at": changed_at}},
            )
            await db.lists.update_many(
                {"user_id": user["id"], "id": {"$in": open_ids}},
                {"$set": {"updated_at": changed_at}},
            )

    prod.update(upd)
    return product_view(prod)


@api.delete("/products/{product_id}", status_code=204)
async def delete_product(product_id: str, user: dict = Depends(get_current_user)):
    """Apaga o produto e o histórico de preços dele. Os itens que já estão em
    listas continuam lá (com o nome que tinham)."""
    prod = await db.products.find_one({"id": product_id, "user_id": user["id"]}, {"_id": 0})
    if not prod:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    await db.price_history.delete_many({"product_id": product_id, "user_id": user["id"]})
    await db.products.delete_one({"id": product_id})
    return None


@api.get("/products/{product_id}/history")
async def product_history(product_id: str, user: dict = Depends(get_current_user)):
    prod = await db.products.find_one({"id": product_id, "user_id": user["id"]}, {"_id": 0})
    if not prod:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    history = await db.price_history.find(
        {"product_id": product_id, "user_id": user["id"]},
        {"_id": 0, "user_id": 0},
    ).sort("date", -1).to_list(200)
    return {"product": product_view(prod), "history": history}


@api.delete("/history/{entry_id}", status_code=204)
async def delete_history_entry(entry_id: str, user: dict = Depends(get_current_user)):
    entry = await db.price_history.find_one({"id": entry_id, "user_id": user["id"]}, {"_id": 0})
    if not entry:
        raise HTTPException(status_code=404, detail="Registro não encontrado")
    await db.price_history.delete_one({"id": entry_id})
    await refresh_product_stats(entry["product_id"])
    return None


# ---------------------------------------------------------------- foto da etiqueta (OCR)
OCR_PROMPT = (
    "Esta é a foto de uma etiqueta de preço de produto em um supermercado brasileiro. "
    "Leia a etiqueta e responda SOMENTE com um JSON, sem texto antes ou depois, neste formato:\n"
    '{"name": string ou null, "price": número ou null, "price_per": "un" | "kg" | "l" | null, '
    '"confidence": "alta" | "media" | "baixa"}\n'
    "Regras: 'price' é o preço de venda em reais, número com ponto decimal (ex.: 6.49), "
    "o preço de maior destaque na etiqueta, não o preço de atacado nem o parcelado. "
    "'price_per' diz se o preço é por unidade, quilo ou litro. "
    "'name' é o nome do produto como aparece na etiqueta, se estiver visível. "
    "Se não conseguir ler o preço com segurança, use price null e confidence baixa."
)


def parse_ocr_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def clean_ocr(data: dict) -> dict:
    price = data.get("price")
    if isinstance(price, str):
        price = price.replace("R$", "").replace(",", ".").strip()
    try:
        price = round(float(price), 2)
        if not (0 < price < 100000):
            price = None
    except (TypeError, ValueError):
        price = None
    name = data.get("name")
    name = name.strip()[:150] if isinstance(name, str) and name.strip() else None
    per = data.get("price_per") if data.get("price_per") in ("un", "kg", "l") else None
    conf = data.get("confidence") if data.get("confidence") in ("alta", "media", "baixa") else "baixa"
    if price is None:
        conf = "baixa"
    return {"name": name, "price": price, "price_per": per, "confidence": conf}


def call_anthropic_vision(image_b64: str, media_type: str) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": OCR_MODEL,
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }],
        },
        timeout=45,
    )
    if resp.status_code != 200:
        logger.error("Anthropic API %s: %s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"status {resp.status_code}")
    blocks = resp.json().get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


@api.post("/ocr/price")
async def ocr_price(body: OcrIn, user: dict = Depends(get_current_user)):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="Leitura de foto ainda não configurada")
    if len(body.image_base64) > OCR_MAX_B64_CHARS:
        raise HTTPException(status_code=413, detail="Foto muito grande, tente novamente")
    quota = ocr_quota(user)
    if quota["remaining"] <= 0:
        raise HTTPException(
            status_code=402,
            detail=f"Você usou as {quota['limit']} leituras de foto deste mês. Digite o preço ou faça upgrade.",
        )
    try:
        text = await run_in_threadpool(call_anthropic_vision, body.image_base64, body.media_type)
    except Exception:
        logger.exception("Falha na leitura da etiqueta")
        raise HTTPException(status_code=502, detail="Não foi possível ler a foto agora. Digite o preço.")
    used = quota["used"] + 1
    await db.users.update_one({"id": user["id"]}, {"$set": {"ocr_month": current_month(), "ocr_count": used}})
    result = clean_ocr(parse_ocr_json(text))
    result["remaining"] = max(quota["limit"] - used, 0)
    return result

@api.get("/admob/ssv")
async def admob_ssv(request: Request):
    raw_query = request.scope["query_string"].decode("ascii")
    if not raw_query:
        raise HTTPException(status_code=400, detail="Callback vazio")


    param_pairs = parse_qsl(raw_query, keep_blank_values=True)
    param_names = [name for name, value in param_pairs]

    if len(param_names) != len(set(param_names)):
        raise HTTPException(
            status_code=400,
            detail="Parâmetros duplicados no callback"
        )

    params = dict(param_pairs)
    key_id = params.get("key_id")
    if not key_id:
        raise HTTPException(status_code=400, detail="ID da chave ausente")

    signature = params.get("signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Assinatura ausente")

    signed_content = raw_query.split("&signature=", 1)[0]
    if signed_content == raw_query or "&key_id=" not in raw_query:
        raise HTTPException(status_code=400, detail="Formato do callback inválido")

    keys = await run_in_threadpool(fetch_admob_public_keys)
    public_key = find_admob_public_key(keys, key_id)
    try:
        verify_admob_signature(signed_content, signature, public_key)
    except (InvalidSignature, ValueError):
        print("ADMOB SSV: assinatura rejeitada", flush=True)
        raise HTTPException(
            status_code=403,
            detail="Assinatura do AdMob inválida"
        )
    if params.get("ad_unit") == "1234567890" and params.get("transaction_id") == "1234567890":
        return {"status": "test_verified"}
    if params.get("ad_unit") != ADMOB_REWARDED_AD_UNIT_ID:
        raise HTTPException(
            status_code=403,
            detail="Unidade de anúncio inválida"
        )
    if params.get("reward_amount") != str(REWARD_ITEMS_PER_AD):
        raise HTTPException(
            status_code=403,
            detail="Quantidade da recompensa inválida"
        )
    if params.get("reward_item") != "produtos_extras":
        raise HTTPException(
            status_code=403,
            detail="Tipo de recompensa inválido"
        )
    timestamp = params.get("timestamp")
    try:
        ad_time = datetime.fromtimestamp(
            int(timestamp) / 1000,
            tz=timezone.utc
        )
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(
            status_code=400,
            detail="Data do anúncio inválida"
        )

    now = datetime.now(timezone.utc)
    if ad_time > now + timedelta(minutes=5) or ad_time < now - timedelta(hours=24):
        raise HTTPException(
            status_code=403,
            detail="Anúncio fora do prazo permitido"
        )
    transaction_id = params.get("transaction_id")
    if not transaction_id:
        raise HTTPException(
            status_code=400,
            detail="ID da transação ausente"
        )
    existing_reward = await db.ad_rewards.find_one(
        {"transaction_id": transaction_id}
   )
    if existing_reward:
        return {"status": "already_processed"}

    admob_user_id = params.get("user_id")
    if not admob_user_id:
        raise HTTPException(
            status_code=400,
            detail="Usuário do anúncio não informado"
        )
    list_id = params.get("custom_data")
    if not list_id:
        raise HTTPException(
            status_code=400,
            detail="Lista do anúncio não informada"
        )
    reward_list = await db.lists.find_one({
        "id": list_id,
        "user_id": admob_user_id
    })

    if not reward_list:
        raise HTTPException(
            status_code=403,
            detail="Lista não pertence ao usuário"
        )
    rewarded_at = datetime.now(timezone.utc)
    expires_at = rewarded_at + timedelta(hours=REWARD_DURATION_HOURS)
    reward_record = {
        "transaction_id": transaction_id,
        "user_id": admob_user_id,
        "list_id": list_id,
        "reward_amount": REWARD_ITEMS_PER_AD,
        "verified": True,
        "created_at": rewarded_at,
        "expires_at": expires_at
    }
    try:
        await db.ad_rewards.insert_one(reward_record)
    except DuplicateKeyError:
        return {"status": "already_processed"}
    return {"status": "verified"}
@api.get("/health")
async def health():
    return {"ok": True}


# ---------------------------------------------------------------- app
app.include_router(api)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    await db.users.create_index("email", unique=True)
    await db.lists.create_index([("user_id", 1), ("updated_at", -1)])
    await db.items.create_index("list_id")
    await db.items.create_index("user_id")
    await db.ad_rewards.create_index([("user_id", 1), ("list_id", 1), ("expires_at", 1)])
    await db.ad_rewards.create_index("transaction_id", unique=True, sparse=True)
    await db.products.create_index([("user_id", 1), ("name_key", 1)], unique=True)
    await db.price_history.create_index([("product_id", 1), ("date", -1)])
    await db.price_history.create_index("item_id")
    await db.price_history.create_index("list_id")


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
