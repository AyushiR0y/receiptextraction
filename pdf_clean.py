import argparse
import base64
import calendar
import difflib
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from zipfile import ZipFile

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

try:
	from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover
	RapidOCR = None

try:
	import easyocr
except Exception:  # pragma: no cover
	easyocr = None

try:
	import pytesseract
except Exception:  # pragma: no cover
	pytesseract = None

try:
	from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover
	convert_from_bytes = None

try:
	import fitz  # PyMuPDF
except Exception:  # pragma: no cover
	fitz = None

try:
	from pypdf import PdfReader
except Exception:  # pragma: no cover
	PdfReader = None


LOGGER = logging.getLogger("receipt_extractor")
RAPIDOCR_ENGINE = None
RAPIDOCR_INIT_FAILED = False
EASYOCR_READER = None
EASYOCR_INIT_FAILED = False
GOOGLE_VISION_API_KEY: Optional[str] = None
OCR_CACHE_DIR = Path(".ocr_cache")
AGENT_PAN_BY_NAME: Dict[str, str] = {}
AGENT_PAN_BY_AADHAAR: Dict[str, str] = {}
AGENT_CODE_BY_NAME: Dict[str, str] = {}  # Maps agent names to agent codes from agentcode.xlsx
AGENT_NAME_BY_CODE: Dict[str, str] = {}  # Maps agent codes to clean agent names from agentcode.xlsx
AGENT_INFO_BY_PAN: Dict[str, Dict[str, str]] = {}  # pan -> info dict (code,name,gstn,...)
AGENT_INFO_BY_CODE: Dict[str, Dict[str, str]] = {}
AZURE_OPENAI_CONFIG_CACHE: Optional[Tuple[str, str, str, str]] = None
AZURE_OPENAI_WORKING_DEPLOYMENT: Optional[str] = None
GOOGLE_VISION_CALL_COUNT = 0
GOOGLE_VISION_DISABLED_FOR_RUN = False
AZURE_AI_CALL_COUNT = 0
AZURE_AI_INPUT_CHARS = 0
AZURE_AI_OUTPUT_CHARS = 0


BANK_PASSWORDS = {
	"axis": "AADCA1701E",
	"axis bank": "AADCA1701E",
	"dbs": "db$126497",
	"marsh": "__Marsh@2025__",
	"marsh india": "__Marsh@2025__",
}


OUTPUT_COLUMNS = [
	"AGENT_CODE",
	"Agent Name",
	"Agent PAN",
	"Name of Service Receipient",
	"BALIC STATE",
	"BALIC GSTN",
	"BROKER GSTN STATE",
	"BROKER GSTN",
	"Vendor Inv Date",
	"Vendor Inv No",
	"Total Inv Amt",
	"BROKERAGE Amount",
	"CGST @ 9%",
	"SGST @ 9%",
	"UTGST",
	"IGST",
	"GST TOTAL AMT",
	"DATE_FROM",
	"DATE_TO",
	"Narration",
	"Type",
	"Micro/Non Micro",
	"SAC Code",
	"Source File",
	"Source Page",
	"Math Valid",
	"Missing Field and Why",
]


@dataclass
class ReceiptLineItem:
	values: Dict[str, str]


GSTIN_PATTERN = r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9]Z[0-9A-Z]"
GSTIN_RE = re.compile(rf"\b({GSTIN_PATTERN})\b", re.IGNORECASE)

GSTIN_STATE_CODE_TO_STATE = {
	"97": "OTHER TERRITORY",
	"01": "Jammu and Kashmir",
	"02": "Himachal Pradesh",
	"03": "Punjab",
	"04": "Chandigarh",
	"05": "Uttarakhand",
	"06": "Haryana",
	"07": "Delhi",
	"08": "Rajasthan",
	"09": "Uttar Pradesh",
	"10": "Bihar",
	"11": "Sikkim",
	"12": "Arunachal Pradesh",
	"13": "Nagaland",
	"14": "Manipur",
	"15": "Mizoram",
	"16": "Tripura",
	"17": "Meghalaya",
	"18": "Assam",
	"19": "Bengal",
	"20": "Jharkhand",
	"21": "Odisha",
	"22": "Chattisgarh",
	"23": "Madhya Pradesh",
	"24": "Gujarat",
	"25": "DAMAN AND DIU",
	"26": "DADRA AND NAGAR HAVELI",
	"27": "Maharashtra",
	"28": "Andhra Pradesh",
	"29": "Karnataka",
	"30": "Goa",
	"31": "Lakshadweep",
	"32": "Kerala",
	"33": "Tamil Nadu",
	"34": "PUDUCHERRY",
	"35": "Andaman and Nicobar Islands",
	"36": "Telangana",
	"37": "Andhra Pradesh",
	"38": "Ladakh",
}

BALIC_PAN = "AADCA1701E"
BALIC_SERVICE_RECIPIENT_NAME = "Bajaj Life Insurance Company"


def normalize_bajaj_company_name(raw_name: str) -> str:
	if not raw_name:
		return ""
	name = re.sub(r"\s+", " ", raw_name).strip()
	name_low = name.lower()
	if "bajaj housing" in name_low:
		return "Bajaj Housing Finance Limited"
	if "bajaj auto finance" in name_low or "bajaj finance" in name_low:
		return "Bajaj Finance Limited"
	if "bajaj auto" in name_low:
		return "Bajaj Housing Finance Limited"
	return name


def extract_bajaj_company_name(text: str) -> str:
	if not text:
		return ""
	patterns = [
		r"(Bajaj\s+Housing\s+Finance\s+(?:Limited|Ltd))",
		r"(Bajaj\s+Auto\s+Finance\s+(?:Limited|Ltd))",
		r"(Bajaj\s+Finance\s+(?:Limited|Ltd))",
		r"(Bajaj\s+Allianz\s+Life\s+Insurance(?:\s+Company)?\s+(?:Limited|Ltd))",
		r"(Bajaj\s+Life\s+Insurance(?:\s+Company)?\s+(?:Limited|Ltd))",
	]
	for pattern in patterns:
		match = re.search(pattern, text, re.IGNORECASE)
		if match:
			return normalize_bajaj_company_name(match.group(1))
	return ""


def normalize_balic_service_recipient_name(raw_name: str, context_text: str = "") -> str:
	candidate = sanitize_party_name(normalize_balic_company_name(raw_name or ""))
	if candidate and candidate.lower().startswith("bajaj") and "auto" not in candidate.lower():
		return candidate
	if context_text:
		extracted = extract_bajaj_company_name(context_text)
		if extracted:
			return extracted
	cleaned = sanitize_party_name(raw_name or "")
	if cleaned and cleaned.lower().startswith("bajaj"):
		return normalize_balic_company_name(cleaned)
	return cleaned

STATE_NAME_ALIASES = {
	"west bengal": "Bengal",
	"chhattisgarh": "Chattisgarh",
	"andaman and nicobar islands": "Andaman and Nicobar",
	"dadra and nagar haveli and daman and diu": "DADRA AND NAGAR HAVELI",
	"puducherry": "PUDUCHERRY",
	"jammu and kashmir": "Jammu and Kashmir",
	"andaman and nicobar": "Andaman and Nicobar",
	"other territory": "OTHER TERRITORY",
	# common OCR/misspelling variants
	"telengana": "Telangana",
	"tamilnadu": "Tamil Nadu",
}


def configure_logging(verbose: bool) -> None:
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(message)s",
	)


def normalize_text(text: str) -> str:
	text = text.replace("\x0c", " ")
	text = re.sub(r"[ \t]+", " ", text)
	text = re.sub(r"\r", "\n", text)
	text = re.sub(r"\n{2,}", "\n", text)
	return text.strip()


def _normalize_agent_name_key(name: str) -> str:
	return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def extract_aadhaar(text: str) -> str:
	if not text:
		return ""
	m = re.search(
		r"(?:aadhaar|aadhar|uid)\s*(?:no|number)?\s*[:\-]?\s*([0-9][0-9\s-]{10,16})",
		text,
		re.IGNORECASE,
	)
	if not m:
		return ""
	digits = re.sub(r"\D", "", m.group(1) or "")
	return digits if len(digits) == 12 else ""


def backfill_agent_pan(fields: Dict[str, str], source_text: str) -> Dict[str, str]:
	result = dict(fields)
	name = (result.get("Agent Name", "") or "").strip()
	pan = (result.get("Agent PAN", "") or "").strip().upper()
	aadhaar = extract_aadhaar(source_text)
	name_key = _normalize_agent_name_key(name)
	code = (result.get("AGENT_CODE", "") or "").strip().upper()

	def _canonical_agent_pan() -> str:
		if code and code in AGENT_INFO_BY_CODE:
			info = AGENT_INFO_BY_CODE[code]
			if info.get("pan"):
				return str(info.get("pan") or "").strip().upper()
		if name_key:
			mapped_code = AGENT_CODE_BY_NAME.get(name_key)
			if mapped_code and mapped_code in AGENT_INFO_BY_CODE:
				info = AGENT_INFO_BY_CODE[mapped_code]
				if info.get("pan"):
					return str(info.get("pan") or "").strip().upper()
			mapped_pan = AGENT_PAN_BY_NAME.get(name_key)
			if mapped_pan and mapped_pan in AGENT_INFO_BY_PAN:
				info = AGENT_INFO_BY_PAN[mapped_pan]
				if info.get("pan"):
					return str(info.get("pan") or "").strip().upper()
		if pan and pan in AGENT_INFO_BY_PAN:
			info = AGENT_INFO_BY_PAN[pan]
			if info.get("pan"):
				return str(info.get("pan") or "").strip().upper()
		return ""

	canonical_pan = _canonical_agent_pan()
	if canonical_pan and (not pan or not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan) or pan == BALIC_PAN or pan != canonical_pan):
		result["Agent PAN"] = canonical_pan
		pan = canonical_pan

	pan_ok = bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan))

	if pan_ok:
		if name_key:
			AGENT_PAN_BY_NAME[name_key] = pan
		if aadhaar:
			AGENT_PAN_BY_AADHAAR[aadhaar] = pan
		# If we have PAN and detailed info, prefer to populate agent code and GSTN too
		info = AGENT_INFO_BY_PAN.get(pan)
		if info:
			if not (result.get("AGENT_CODE", "") or "").strip() and info.get("code"):
				result["AGENT_CODE"] = info.get("code")
			if not (result.get("BROKER GSTN", "") or "").strip() and info.get("gstn"):
				result["BROKER GSTN"] = info.get("gstn")
		return result

	# Prefer Aadhaar mapping first (stronger identity), then agent-name mapping.
	if aadhaar and aadhaar in AGENT_PAN_BY_AADHAAR:
		result["Agent PAN"] = AGENT_PAN_BY_AADHAAR[aadhaar]
	elif name_key and name_key in AGENT_PAN_BY_NAME:
		result["Agent PAN"] = AGENT_PAN_BY_NAME[name_key]
	else:
		# Try to populate PAN from agent code details if agent code is present or can be inferred
		code_from_name = AGENT_CODE_BY_NAME.get(name_key)
		code = (result.get("AGENT_CODE", "") or "").strip() or (code_from_name or "")
		if code:
			info = AGENT_INFO_BY_CODE.get(code)
			if info and info.get("pan"):
				result["Agent PAN"] = info.get("pan")

	# If PAN now present, try to backfill agent code and broker GSTN from loaded agent details
	pan_now = (result.get("Agent PAN", "") or "").strip().upper()
	if pan_now and pan_now in AGENT_INFO_BY_PAN:
		info = AGENT_INFO_BY_PAN[pan_now]
		if not (result.get("AGENT_CODE", "") or "").strip() and info.get("code"):
			result["AGENT_CODE"] = info.get("code")
		if not (result.get("BROKER GSTN", "") or "").strip() and info.get("gstn"):
			result["BROKER GSTN"] = info.get("gstn")

	# Finally, if agent code exists but no PAN, try to pull from code mapping
	existing_code = (result.get("AGENT_CODE", "") or "").strip()
	if existing_code and existing_code in AGENT_INFO_BY_CODE:
		info = AGENT_INFO_BY_CODE[existing_code]
		if not (result.get("Agent PAN", "") or "").strip() and info.get("pan"):
			result["Agent PAN"] = info.get("pan")
		if not (result.get("BROKER GSTN", "") or "").strip() and info.get("gstn"):
			result["BROKER GSTN"] = info.get("gstn")

	return result


def load_agent_codes_from_xlsx(xlsx_path: str = "agentcode.xlsx") -> None:
	"""Load agent code mappings from agentcode.xlsx file.
	
	Populates:
	- AGENT_CODE_BY_NAME: normalized agent name -> agent code
	- AGENT_NAME_BY_CODE: agent code -> clean agent name
	"""
	global AGENT_CODE_BY_NAME, AGENT_NAME_BY_CODE
	AGENT_CODE_BY_NAME.clear()
	AGENT_NAME_BY_CODE.clear()
	
	try:
		agent_code_path = Path(xlsx_path)
		if not agent_code_path.exists():
			LOGGER.warning("Agent code file not found: %s", xlsx_path)
			return
		# Try to read all sheets; primary sheet contains AGENT_CODE/Name, optional second sheet contains full details including PAN/GSTN
		sheets = pd.read_excel(agent_code_path, sheet_name=None)
		# Primary: first sheet (or sheet named 'Sheet1'/'agent_codes')
		first_sheet = None
		for name, df in sheets.items():
			first_sheet = df
			break
		if first_sheet is not None:
			for _, row in first_sheet.iterrows():
				code = str(row.get("AGENT_CODE", "") or row.get("Agent Code", "") or "").strip()
				name = str(row.get("Name", "") or row.get("Agent Name", "") or "").strip()
				if code and name:
					name_key = _normalize_agent_name_key(name)
					AGENT_CODE_BY_NAME[name_key] = code
					AGENT_NAME_BY_CODE[code] = name
					LOGGER.debug("Loaded agent code: %s -> %s", code, name)

		# Optional full-details sheet: look for a sheet named 'agent_details_full' or the second sheet
		details_sheet = None
		if "agent_details_full" in {k.lower(): k for k in sheets.keys()}:
			details_sheet = sheets[[k for k in sheets.keys() if k.lower() == "agent_details_full"][0]]
		elif len(sheets) > 1:
			# take second sheet
			keys = list(sheets.keys())
			details_sheet = sheets[keys[1]]
		
		if details_sheet is not None:
			for _, row in details_sheet.iterrows():
				code = str(row.get("AGENT_CODE", "") or row.get("Agent Code", "") or "").strip()
				name = str(row.get("Name", "") or row.get("Agent Name", "") or "").strip()
				pan = str(row.get("PAN", "") or row.get("Agent PAN", "") or "").strip().upper()
				gstn = str(row.get("GSTN", "") or row.get("Agent GSTN", "") or "").strip().upper()
				state = str(row.get("State", "") or "").strip()
				addr = str(row.get("Address", "") or "").strip()
				info = {"code": code, "name": name, "pan": pan, "gstn": gstn, "state": state, "address": addr}
				if pan and re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan, re.IGNORECASE):
					AGENT_INFO_BY_PAN[pan] = info
				if code:
					AGENT_INFO_BY_CODE[code] = info
				# Also seed AGENT_PAN_BY_NAME to help backfill
				if name and pan:
					AGENT_PAN_BY_NAME[_normalize_agent_name_key(name)] = pan
				LOGGER.debug("Loaded agent detail: %s -> PAN %s GSTN %s", name, pan, gstn)
		
		LOGGER.info("Loaded %d agent codes from %s", len(AGENT_CODE_BY_NAME), xlsx_path)
	except Exception as exc:
		LOGGER.warning("Failed to load agent codes from %s: %s", xlsx_path, exc)


def find_best_matching_agent_code(extracted_name: str, similarity_threshold: float = 0.6) -> Tuple[Optional[str], Optional[str], float]:
	"""Find best matching agent code and clean name using similarity matching.
	
	Args:
		extracted_name: Agent name extracted from invoice
		similarity_threshold: Minimum similarity ratio to consider a match
	
	Returns:
		Tuple of (agent_code, clean_agent_name, similarity_ratio)
		Returns (None, None, 0) if no match found
	"""
	if not extracted_name or not AGENT_CODE_BY_NAME:
		return None, None, 0.0
	
	extracted_key = _normalize_agent_name_key(extracted_name)
	if not extracted_key:
		return None, None, 0.0
	
	# First try exact match in normalized form
	if extracted_key in AGENT_CODE_BY_NAME:
		code = AGENT_CODE_BY_NAME[extracted_key]
		clean_name = AGENT_NAME_BY_CODE.get(code, extracted_name)
		return code, clean_name, 1.0
	
	# Try fuzzy matching using difflib
	best_match_key = None
	best_ratio = 0.0
	
	for agent_name_key in AGENT_CODE_BY_NAME.keys():
		ratio = difflib.SequenceMatcher(None, extracted_key, agent_name_key).ratio()
		if ratio > best_ratio:
			best_ratio = ratio
			best_match_key = agent_name_key
	
	# Return match only if similarity is above threshold
	if best_match_key and best_ratio >= similarity_threshold:
		code = AGENT_CODE_BY_NAME[best_match_key]
		clean_name = AGENT_NAME_BY_CODE.get(code, extracted_name)
		return code, clean_name, best_ratio
	
	return None, None, best_ratio


def load_env_file_if_present() -> None:
	"""Load simple KEY=VALUE pairs from .env into process environment if missing."""
	env_path = Path(".env")
	if not env_path.exists():
		return
	try:
		for raw_line in env_path.read_text(encoding="utf-8").splitlines():
			line = raw_line.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip().strip('"').strip("'")
			if key and key not in os.environ:
				os.environ[key] = value
	except Exception as exc:
		LOGGER.warning("Failed to read .env file: %s", exc)


def get_google_vision_api_key() -> str:
	global GOOGLE_VISION_API_KEY, GOOGLE_VISION_DISABLED_FOR_RUN
	if GOOGLE_VISION_API_KEY is not None:
		return GOOGLE_VISION_API_KEY
	if GOOGLE_VISION_DISABLED_FOR_RUN:
		return ""

	load_env_file_if_present()
	GOOGLE_VISION_API_KEY = (os.getenv("GOOGLE_VISION_API_KEY") or "").strip()
	if GOOGLE_VISION_API_KEY:
		LOGGER.info("Google Vision OCR is enabled for low-confidence images/pages")
	return GOOGLE_VISION_API_KEY


def allow_arithmetic_autofill() -> bool:
	# Disabled by default per user requirement: only values directly read from PDFs.
	raw = (os.getenv("ARITHMETIC_AUTOFILL") or "0").strip().lower()
	return raw in {"1", "on", "true", "yes"}


def get_azure_openai_config() -> Optional[Tuple[str, str, str, str]]:
	"""Return (endpoint, api_key, deployment, api_version) when Azure OpenAI is configured."""
	global AZURE_OPENAI_CONFIG_CACHE
	if AZURE_OPENAI_CONFIG_CACHE is not None:
		return AZURE_OPENAI_CONFIG_CACHE

	load_env_file_if_present()
	endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
	api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
	deployment = (
		(os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI") or "").strip()
		or (os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT_NAME") or "").strip()
		or (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()
		or (os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or "").strip()
	)
	if not deployment:
		deployment = "gpt-4o-mini"
	api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21").strip()

	if not endpoint or not api_key:
		AZURE_OPENAI_CONFIG_CACHE = None
		return None

	AZURE_OPENAI_CONFIG_CACHE = (endpoint, api_key, deployment, api_version)
	LOGGER.info("Azure OpenAI enabled with deployment: %s", deployment)
	return AZURE_OPENAI_CONFIG_CACHE


def _normalize_key(key: str) -> str:
	return re.sub(r"[^a-z0-9]", "", (key or "").lower())


def is_valid_gstin(value: str) -> bool:
	return bool(GSTIN_RE.fullmatch((value or "").strip().upper()))


def extract_gstin_candidates(text: str) -> List[str]:
	seen: Set[str] = set()
	candidates: List[str] = []
	for match in GSTIN_RE.finditer(text or ""):
		candidate = match.group(1).upper()
		if candidate in seen:
			continue
		seen.add(candidate)
		candidates.append(candidate)
	return candidates


def gstin_to_state_name(value: str) -> str:
	candidate = (value or "").strip().upper()
	if not is_valid_gstin(candidate):
		return ""
	return GSTIN_STATE_CODE_TO_STATE.get(candidate[:2], "")


def state_name_to_state_code(state_name: str) -> str:
	needle = (state_name or "").strip().lower()
	if not needle:
		return ""
	needle = STATE_NAME_ALIASES.get(needle, state_name or "").strip().lower()
	for code, name in GSTIN_STATE_CODE_TO_STATE.items():
		if (name or "").strip().lower() == needle:
			return code
	return ""


def compute_gstin_check_digit(first_14: str) -> str:
	"""Compute GSTIN check digit for the first 14 characters using GSTN base-36 checksum."""
	alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	char_to_index = {ch: idx for idx, ch in enumerate(alphabet)}
	if not re.fullmatch(r"[0-9A-Z]{14}", (first_14 or "").strip().upper()):
		return ""
	payload = (first_14 or "").strip().upper()
	total = 0
	for i, ch in enumerate(payload):
		code_point = char_to_index[ch]
		factor = 1 if (i % 2 == 0) else 2
		product = code_point * factor
		total += (product // 36) + (product % 36)
	check_code_point = (36 - (total % 36)) % 36
	return alphabet[check_code_point]


def synthesize_gstin_from_pan_and_state(agent_pan: str, state_code: str) -> str:
	pan = (agent_pan or "").strip().upper()
	code = (state_code or "").strip()
	if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan):
		return ""
	if not re.fullmatch(r"[0-9]{2}", code):
		return ""
	first_14 = f"{code}{pan}1Z"
	check_digit = compute_gstin_check_digit(first_14)
	if not check_digit:
		return ""
	candidate = f"{first_14}{check_digit}"
	return candidate if is_valid_gstin(candidate) else ""


def state_from_invoice_prefix(invoice_no: str) -> str:
	prefix_match = re.match(r"^\s*(\d{2})", (invoice_no or ""))
	if not prefix_match:
		return ""
	return GSTIN_STATE_CODE_TO_STATE.get(prefix_match.group(1), "")


def state_from_source_identifier(source_text: str) -> str:
	text = str(source_text or "")
	patterns = [
		r"(?:^|[^0-9])(\d{2})\d{10,}(?:[^0-9]|$)",
		r"(?:^|[_/-])(\d{2})010226",  # common City Union/Bajaj zip invoice-like IDs
	]
	for pattern in patterns:
		match = re.search(pattern, text)
		if not match:
			continue
		state = GSTIN_STATE_CODE_TO_STATE.get(match.group(1), "")
		if state:
			return state
	return ""


def infer_state_hint(source_path: Optional[Path], context_text: str = "") -> str:
	"""Infer a likely state name from the source path or OCR context."""
	text = f"{str(source_path or '')} {context_text or ''}".lower()
	if any(tok in text for tok in ("tamil nadu", "tamilnadu", "chennai", "coimbatore", "madurai")):
		return "Tamil Nadu"
	if any(tok in text for tok in ("telangana", "telengana", "hyderabad", "secunderabad")):
		return "Telangana"
	if source_path:
		inferred = state_from_source_identifier(str(source_path))
		if inferred:
			return inferred
	return ""


def apply_company_wise_issue_mapping(row: Dict[str, str], source_path: Optional[Path], context_text: str = "") -> Dict[str, str]:
	result = dict(row)
	source_low = str(source_path).lower() if source_path else ""
	ctx = context_text or ""
	recipient_low = (result.get("Name of Service Receipient", "") or "").strip().lower()

	company_signals = [
		"city union bank",
		"dhanlaxmi bank",
		"jammu and kashmir bank",
		"turtlemint",
		"capri global",
		"bajaj allianz",
		"bajaj life",
	]
	# Check document text (ctx) instead of folder name (source_path) to avoid false positives
	is_issue_company = any(sig in ctx.lower() for sig in company_signals)
	is_bajaj = (
		"bajaj allianz life insurance company ltd" in recipient_low
		or any(sig in source_low for sig in ["bajaj allianz", "bajaj life", "bajaj.zip", "-bajaj"])
		or bool(re.search(r"bajaj\s+allianz|bajaj\s+life", ctx, re.IGNORECASE))
	)

	# Prioritize state extraction from GSTINs over unreliable invoice prefix.
	balic_gstn = (result.get("BALIC GSTN", "") or "").strip().upper()

	broker_gstn = (result.get("BROKER GSTN", "") or "").strip().upper()
	balic_state = gstin_to_state_name(balic_gstn) if balic_gstn else ""
	broker_state = gstin_to_state_name(broker_gstn) if broker_gstn else ""
	if not balic_state:
		balic_state = state_from_invoice_prefix(result.get("Vendor Inv No", ""))
	if not balic_state and source_path:
		balic_state = state_from_source_identifier(str(source_path))
	if not broker_state and balic_state:
		broker_state = balic_state
	if is_issue_company:
		if balic_state and not (result.get("BALIC STATE", "") or "").strip():
			result["BALIC STATE"] = balic_state
		if broker_state and not (result.get("BROKER GSTN STATE", "") or "").strip():
			result["BROKER GSTN STATE"] = broker_state
		# If GSTINs are missing but the state is known, synthesize from PAN + state code.
		state_code = state_name_to_state_code(balic_state)
		if state_code and not is_valid_gstin((result.get("BALIC GSTN", "") or "").strip().upper()):
			synth_balic = synthesize_gstin_from_pan_and_state(BALIC_PAN, state_code)
			if synth_balic:
				result["BALIC GSTN"] = synth_balic
				if not (result.get("BALIC STATE", "") or "").strip():
					result["BALIC STATE"] = balic_state
				state_code = state_name_to_state_code(result.get("BROKER GSTN STATE", "") or balic_state)
		if state_code and not is_valid_gstin((result.get("BROKER GSTN", "") or "").strip().upper()):
			agent_pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", ctx, re.IGNORECASE)
			agent_pan_hint = (result.get("Agent PAN", "") or "").strip().upper() or (agent_pan_match.group(1).upper() if agent_pan_match else "")
			if agent_pan_hint and agent_pan_hint != BALIC_PAN:
				synth_broker = synthesize_gstin_from_pan_and_state(agent_pan_hint, state_code)
				if synth_broker:
					result["BROKER GSTN"] = synth_broker
					if not (result.get("BROKER GSTN STATE", "") or "").strip():
						result["BROKER GSTN STATE"] = gstin_to_state_name(synth_broker)
	invoice_state = balic_state
	if broker_gstn and not is_valid_gstin(broker_gstn):
		if re.search(r"BAJAJRETAIL|RET\d{3,}|^3664BAJAJ|^R\d[A-Z0-9]{10,}$", broker_gstn, re.IGNORECASE):
			result["BROKER GSTN"] = ""

	# For Bajaj: aggressive GSTIN recovery. If broker GSTN is blank, try to fill from candidates.
	# Known Bajaj GSTINs to whitelist for proper identification
	known_bajaj_gstins = {"27AADCA1701E1ZD", "29AADCA1701E1Z9", "33AADCA1701E1ZK"}
	if is_bajaj:
		current_broker = (result.get("BROKER GSTN", "") or "").strip().upper()
		if not current_broker or not is_valid_gstin(current_broker):
			candidates = extract_gstin_candidates(ctx)
			# Filter to known Bajaj GSTINs first; fallback to other state-matching candidates
			balic_gstn = (result.get("BALIC GSTN", "") or "").strip().upper()
			bajaj_candidates = [c for c in candidates if c in known_bajaj_gstins]
			if not bajaj_candidates:
				# Fallback: use any valid GSTIN except BALIC
				bajaj_candidates = [c for c in candidates if is_valid_gstin(c) and c != balic_gstn]
			if bajaj_candidates:
				# Prefer known Bajaj GSTIN, then state-matching, then first valid
				preferred = ""
				for cand in bajaj_candidates:
					if cand in known_bajaj_gstins:
						preferred = cand
						break
				if not preferred and invoice_state:
					for cand in bajaj_candidates:
						if gstin_to_state_name(cand) == invoice_state:
							preferred = cand
							break
				if not preferred:
					preferred = bajaj_candidates[0]
				result["BROKER GSTN"] = preferred
				if not (result.get("BROKER GSTN STATE", "") or "").strip():
					result["BROKER GSTN STATE"] = gstin_to_state_name(preferred)

	# For issue-heavy company layouts, prefer broker GSTIN that matches Agent PAN when available.
	agent_pan = (result.get("Agent PAN", "") or "").strip().upper()
	if is_issue_company and re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", agent_pan):
		current_broker = (result.get("BROKER GSTN", "") or "").strip().upper()
		if not is_valid_gstin(current_broker) or (BALIC_PAN in current_broker):
			candidates = extract_gstin_candidates(ctx)
			preferred = ""
			for candidate in candidates:
				if agent_pan in candidate:
					if invoice_state and gstin_to_state_name(candidate) == invoice_state:
						preferred = candidate
						break
					if not preferred:
						preferred = candidate
			if preferred:
				result["BROKER GSTN"] = preferred
				if not (result.get("BROKER GSTN STATE", "") or "").strip():
					result["BROKER GSTN STATE"] = gstin_to_state_name(preferred)

	# If GSTINs are still missing for issue companies, use state hints from source/context.
	if is_issue_company:
		state_hint = infer_state_hint(source_path, ctx)
		state_code = state_name_to_state_code(state_hint)
		if state_hint and not (result.get("BALIC STATE", "") or "").strip():
			result["BALIC STATE"] = state_hint
		if state_code and not (result.get("BALIC GSTN", "") or "").strip():
			synth_balic = synthesize_gstin_from_pan_and_state(BALIC_PAN, state_code)
			if synth_balic:
				result["BALIC GSTN"] = synth_balic
				result["BALIC STATE"] = gstin_to_state_name(synth_balic) or state_hint
		if state_code and not (result.get("BROKER GSTN", "") or "").strip() and agent_pan:
			synth_broker = synthesize_gstin_from_pan_and_state(agent_pan, state_code)
			if synth_broker:
				result["BROKER GSTN"] = synth_broker
				result["BROKER GSTN STATE"] = gstin_to_state_name(synth_broker)

	return result


def _strip_trailing_context_noise(name: str) -> str:
	name = re.sub(r"\s+", " ", (name or "")).strip(" ,:-")
	name = re.sub(
		r"\s+(?:\d{1,2}\s*[-/]\s*\d{1,2}(?:\s*[A-Za-z]{3,9}\s*['’\-]?\s*\d{2,4})?|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*['’\-]?\s*\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s*['’\-]?\s*\d{2,4}|(?:outwardinvoice|invoice|commission|billing|billings|receipt|pdf))\s*$",
		"",
		name,
		flags=re.IGNORECASE,
	)
	name = re.sub(r"\s+\d{1,2}$", "", name).strip(" ,:-")
	return name


def _extract_labeled_gstin(patterns: Sequence[str], text: str) -> str:
	for pattern in patterns:
		match = re.search(pattern, text or "", re.IGNORECASE)
		if not match:
			continue
		candidate = (match.group(1) or "").strip().upper()
		if is_valid_gstin(candidate):
			return candidate
	return ""


BALIC_GSTIN_WHITELIST: Set[str] = {
	"27AADCA1701E1ZD",
	"29AADCA1701E1Z9",
	"33AADCA1701E1ZK",
	"07AADCA1701E2ZE",
	"03AADCA1701E1ZN",
	"08AADCA1701E1ZD",
	"36AADCA1701E1ZE",
	"21AADCA1701E1ZP",
	"19AADCA1701E1ZA",
	
}


def choose_balic_and_broker_gstin(flattened: str) -> Tuple[str, str]:
	gstin_values = extract_gstin_candidates(flattened)
	pan_in_text = ""
	pan_match = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", flattened or "", re.IGNORECASE)
	if pan_match:
		pan_in_text = (pan_match.group(1) or "").upper()
	balic_pan_hits = [g for g in gstin_values if gstin_to_pan(g) == BALIC_PAN]
	if balic_pan_hits:
		balic_gstn = balic_pan_hits[0]
		broker_gstn = ""
		for candidate in gstin_values:
			if candidate != balic_gstn and gstin_to_pan(candidate) != BALIC_PAN:
				broker_gstn = candidate
				break
		if not is_valid_gstin(broker_gstn):
			broker_gstn = ""
		return balic_gstn, broker_gstn
	invoice_state = gstin_to_state_name(gstin_values[0]) if gstin_values else ""
	# Primary rule: if whitelist GSTIN exists, it is BALIC by definition.
	whitelist_hits = [g for g in gstin_values if g in BALIC_GSTIN_WHITELIST]
	if whitelist_hits:
		balic_gstn = whitelist_hits[0]
		broker_gstn = ""
		for candidate in gstin_values:
			if candidate != balic_gstn and candidate not in BALIC_GSTIN_WHITELIST:
				broker_gstn = candidate
				break
		if not is_valid_gstin(broker_gstn):
			broker_gstn = ""
		return balic_gstn, broker_gstn

	balic_gstn = _extract_labeled_gstin(
		[
			rf"balic\s*gst(?:n|in)\s*[:\-]?\s*({GSTIN_PATTERN})",
			rf"bajaj\s+allianz[^\n]{0,80}?gst(?:n|in)\s*[:\-]?\s*({GSTIN_PATTERN})",
			rf"service\s*recipient[^\n]{0,80}?gst(?:n|in)\s*[:\-]?\s*({GSTIN_PATTERN})",
			rf"(?:billed\s*to|bill\s*to|customer|insured)[^\n]{0,100}?gst(?:n|in)\s*[:\-]?\s*({GSTIN_PATTERN})",
		],
		flattened,
	)
	broker_gstn = _extract_labeled_gstin(
		[
			rf"broker\s*gst(?:n|in)\s*[:\-]?\s*({GSTIN_PATTERN})",
			rf"corporate\s*agent[^\n]{0,80}?gst(?:n|in)\s*[:\-]?\s*({GSTIN_PATTERN})",
			rf"supplier[^\n]{0,80}?gst(?:n|in)\s*[:\-]?\s*({GSTIN_PATTERN})",
			rf"service\s*provider[^\n]{0,80}?gst(?:n|in)\s*[:\-]?\s*({GSTIN_PATTERN})",
			rf"agent[^\n]{0,80}?gst(?:n|in)\s*[:\-]?\s*({GSTIN_PATTERN})",
		],
		flattened,
	)

	# Document-level label proximity fallback when explicit label extract misses.
	if (not balic_gstn) or (not broker_gstn):
		for m in re.finditer(GSTIN_PATTERN, flattened or "", re.IGNORECASE):
			candidate = (m.group(0) or "").strip().upper()
			if not is_valid_gstin(candidate):
				continue
			window = (flattened[max(0, m.start() - 90): min(len(flattened), m.end() + 90)] or "").lower()
			if (not balic_gstn) and re.search(r"\b(?:recipient|service\s*recipient|billed\s*to|bill\s*to|customer|insured)\b", window):
				balic_gstn = candidate
			if (not broker_gstn) and re.search(r"\b(?:supplier|service\s*provider|broker|corporate\s*agent|agent)\b", window):
				broker_gstn = candidate

	if not balic_gstn:
		for candidate in gstin_values:
			if candidate in BALIC_GSTIN_WHITELIST:
				balic_gstn = candidate
				break

	if not balic_gstn:
		for candidate in gstin_values:
			if "AADCA1701E" in candidate and candidate in BALIC_GSTIN_WHITELIST:
				balic_gstn = candidate
				break

	if not balic_gstn:
		for candidate in gstin_values:
			if "AADCA1701E" in candidate:
				balic_gstn = candidate
				break

	if not broker_gstn:
		for candidate in gstin_values:
			if pan_in_text and pan_in_text in candidate and candidate != balic_gstn:
				broker_gstn = candidate
				break

	if not broker_gstn:
		for candidate in gstin_values:
			if candidate != balic_gstn and candidate not in BALIC_GSTIN_WHITELIST:
				broker_gstn = candidate
				break

	if broker_gstn == balic_gstn:
		broker_gstn = ""

	if not is_valid_gstin(balic_gstn):
		balic_gstn = ""
	if not is_valid_gstin(broker_gstn):
		broker_gstn = ""

	return balic_gstn, broker_gstn


def build_missing_field_reason(row: Dict[str, str], default_reason: str = "") -> str:
	missing: List[str] = []
	if not (row.get("Agent Name", "") or "").strip():
		missing.append("Agent Name")
	if not (row.get("AGENT_CODE", "") or "").strip():
		missing.append("AGENT_CODE")
	if not (row.get("Name of Service Receipient", "") or "").strip():
		missing.append("Name of Service Receipient")
	if not (row.get("Vendor Inv Date", "") or "").strip():
		missing.append("Vendor Inv Date")

	inv_no = clean_invoice_no(row.get("Vendor Inv No", ""))
	if not is_valid_invoice_no(inv_no):
		missing.append("Vendor Inv No")

	brok = _amount_to_float(row.get("BROKERAGE Amount", ""))
	gst_total = _amount_to_float(row.get("GST TOTAL AMT", ""))
	cgst = _amount_to_float(row.get("CGST @ 9%", ""))
	sgst = _amount_to_float(row.get("SGST @ 9%", ""))
	utgst = _amount_to_float(row.get("UTGST", ""))
	igst = _amount_to_float(row.get("IGST", ""))

	if brok <= 0:
		missing.append("BROKERAGE Amount")
	if gst_total <= 0:
		missing.append("GST TOTAL AMT")
	if not (row.get("BALIC GSTN", "") or "").strip():
		missing.append("BALIC GSTN")
	if not (row.get("BROKER GSTN", "") or "").strip():
		missing.append("BROKER GSTN")
	if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", (row.get("Agent PAN", "") or "").strip(), re.IGNORECASE):
		missing.append("Agent PAN")

	state_component = sgst if sgst > 0 else utgst
	if igst <= 0 and cgst <= 0 and state_component <= 0:
		missing.append("IGST/CGST/SGST")
	elif igst <= 0 and (cgst > 0 or state_component > 0):
		if cgst <= 0:
			missing.append("CGST @ 9%")
		if state_component <= 0:
			missing.append("SGST @ 9%")
	elif igst > 0:
		# IGST mode is complete without CGST/SGST.
		pass

	parts: List[str] = []
	if missing:
		parts.append("Missing: " + ", ".join(missing))
	if default_reason:
		parts.append(default_reason)

	return " | ".join(p for p in parts if p) if parts else "OK"


def build_placeholder_row(source_file: str, source_page: str, reason: str) -> ReceiptLineItem:
	values = {col: "" for col in OUTPUT_COLUMNS}
	values["Source File"] = source_file
	values["Source Page"] = source_page
	values["Math Valid"] = "NO: Missing monetary values"
	values["Missing Field and Why"] = reason
	values["Narration"] = ""
	return ReceiptLineItem(values=values)


def get_azure_openai_deployment_candidates(preferred: str) -> List[str]:
	candidates = [
		(os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI") or "").strip(),
		(os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT_NAME") or "").strip(),
		"gpt-4o-mini",
		(preferred or "").strip(),
		(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or "").strip(),
		(os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip(),
	]
	seen: Set[str] = set()
	ordered: List[str] = []
	for dep in candidates:
		if not dep or dep in seen:
			continue
		seen.add(dep)
		ordered.append(dep)
	return ordered


def _ai_row_to_output_fields(ai_row: Dict[str, object]) -> Dict[str, str]:
	alias_to_col = {
		"agentname": "Agent Name",
		"agentpan": "Agent PAN",
		"pan": "Agent PAN",
		"servicerecipient": "Name of Service Receipient",
		"nameofservicereceipient": "Name of Service Receipient",
		"nameofservicerecipient": "Name of Service Receipient",
		"recipient": "Name of Service Receipient",
		"particulars": "Narration",
		"description": "Narration",
		"balicgstn": "BALIC GSTN",
		"balicgstin": "BALIC GSTN",
		"brokergstin": "BROKER GSTN",
		"brokergstn": "BROKER GSTN",
		"brokergstnstate": "BROKER GSTN STATE",
		"brokergstinstate": "BROKER GSTN STATE",
		"brokerstate": "BROKER GSTN STATE",
		"state": "BALIC STATE",
		"balicstate": "BALIC STATE",
		"date": "Vendor Inv Date",
		"vendorinvdate": "Vendor Inv Date",
		"invoicedate": "Vendor Inv Date",
		"invdate": "Vendor Inv Date",
		"invoicenumber": "Vendor Inv No",
		"invoiceno": "Vendor Inv No",
		"billno": "Vendor Inv No",
		"billnumber": "Vendor Inv No",
		"vendorinvno": "Vendor Inv No",
		"invoiceamount": "Total Inv Amt",
		"totalinvamt": "Total Inv Amt",
		"totalamount": "Total Inv Amt",
		"netamount": "Total Inv Amt",
		"amount": "Total Inv Amt",
		"taxableamount": "BROKERAGE Amount",
		"brokerageamount": "BROKERAGE Amount",
		"commissionamount": "BROKERAGE Amount",
		"cgst": "CGST @ 9%",
		"cgstamount": "CGST @ 9%",
		"sgst": "SGST @ 9%",
		"sgstamount": "SGST @ 9%",
		"igst": "IGST",
		"igstamount": "IGST",
		"gsttotalamount": "GST TOTAL AMT",
		"gsttotalamt": "GST TOTAL AMT",
		"gstawt": "GST TOTAL AMT",
		"gstamount": "GST TOTAL AMT",
		"narration": "Narration",
	}

	result: Dict[str, str] = {col: "" for col in OUTPUT_COLUMNS}
	for raw_key, raw_value in ai_row.items():
		col = alias_to_col.get(_normalize_key(str(raw_key)), "")
		if not col:
			continue
		value = "" if raw_value is None else str(raw_value).strip()
		if col in {"Total Inv Amt", "CGST @ 9%", "SGST @ 9%", "IGST", "GST TOTAL AMT"}:
			value = clean_amount(value)
		elif col == "Vendor Inv Date":
			value = try_parse_date(value)
		elif col == "BALIC STATE":
			value = normalize_state_name(value)
		elif col == "Agent PAN":
			value = value.upper()
		result[col] = value

	if not (result.get("Narration", "") or "").strip():
		result["Narration"] = "COMMISSION"

	pan = (result.get("Agent PAN", "") or "").strip().upper()
	balic_gstn = (result.get("BALIC GSTN", "") or "").strip().upper()
	broker_gstn = (result.get("BROKER GSTN", "") or "").strip().upper()
	balic_pan = gstin_to_pan(balic_gstn)
	broker_pan = gstin_to_pan(broker_gstn)

	if pan == BALIC_PAN and broker_pan and broker_pan != BALIC_PAN:
		result["Agent PAN"] = broker_pan
		pan = broker_pan
	elif not pan and broker_pan and broker_pan != BALIC_PAN:
		result["Agent PAN"] = broker_pan
		pan = broker_pan

	if pan and (not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan) or pan == BALIC_PAN):
		result["Agent PAN"] = ""

	for gstin_col in ["BALIC GSTN", "BROKER GSTN"]:
		gstin_val = (result.get(gstin_col, "") or "").strip().upper()
		if gstin_val and not is_valid_gstin(gstin_val):
			result[gstin_col] = ""
		elif gstin_val:
			result[gstin_col] = gstin_val

	if (result.get("BALIC GSTN", "") or "").strip():
		result["BALIC STATE"] = gstin_to_state_name(result.get("BALIC GSTN", ""))

	if (result.get("BROKER GSTN", "") or "").strip():
		result["BROKER GSTN STATE"] = gstin_to_state_name(result.get("BROKER GSTN", ""))

	inv = (result.get("Vendor Inv No", "") or "").strip()
	if inv and not is_valid_invoice_no(inv):
		result["Vendor Inv No"] = ""

	return result


def _choose_tax_mode(brok: float, igst: float, state_tax: float, gst_total: float, cgst: float, sgst_or_utgst: float) -> str:
	"""Choose one coherent tax mode when OCR populates both IGST and CGST/SGST."""
	if igst <= 0 and state_tax <= 0:
		return "none"
	if igst > 0 and state_tax <= 0:
		return "igst"
	if state_tax > 0 and igst <= 0:
		return "state"

	# Both are present: prefer component closest to expected tax from brokerage.
	if brok > 0:
		expected = brok * 0.18
		if abs(igst - expected) <= abs(state_tax - expected):
			return "igst"
		return "state"

	# Fallback to whichever aligns better with GST total when brokerage is missing.
	if gst_total > 0:
		if abs(igst - gst_total) <= abs(state_tax - gst_total):
			return "igst"
		return "state"

	# Last fallback: if CGST and SGST/UTGST are close mirrors, prefer state-tax mode.
	if cgst > 0 and sgst_or_utgst > 0 and abs(cgst - sgst_or_utgst) <= max(cgst, sgst_or_utgst) * 0.15:
		return "state"
	return "igst" if igst >= state_tax else "state"


def _fmt_amount(value: float) -> str:
	return f"{value:.2f}".rstrip("0").rstrip(".")


def _tax_mode_summary(row: Dict[str, str]) -> str:
	cgst = _amount_to_float(row.get("CGST @ 9%", ""))
	sgst = _amount_to_float(row.get("SGST @ 9%", ""))
	utgst = _amount_to_float(row.get("UTGST", ""))
	igst = _amount_to_float(row.get("IGST", ""))
	parts: List[str] = []
	if igst > 0:
		parts.append(f"IGST={_fmt_amount(igst)}")
	if cgst > 0:
		parts.append(f"CGST={_fmt_amount(cgst)}")
	if sgst > 0:
		parts.append(f"SGST={_fmt_amount(sgst)}")
	if utgst > 0:
		parts.append(f"UTGST={_fmt_amount(utgst)}")
	if not parts:
		return "no clear GST tax components were extracted"
	if igst > 0 and cgst <= 0 and sgst <= 0 and utgst <= 0:
		return "IGST-only tax mode"
	if cgst > 0 and (sgst > 0 or utgst > 0) and igst <= 0:
		return "CGST + SGST/UTGST tax mode"
	return "mixed or partial tax components: " + ", ".join(parts)


def _detect_clear_tax_mode(row: Dict[str, str]) -> str:
	"""Return igst/state only when tax mode is explicitly clear from extracted values."""
	cgst = _amount_to_float(row.get("CGST @ 9%", ""))
	sgst = _amount_to_float(row.get("SGST @ 9%", ""))
	utgst = _amount_to_float(row.get("UTGST", ""))
	igst = _amount_to_float(row.get("IGST", ""))

	has_igst = igst > 0
	has_state = cgst > 0 and (sgst > 0 or utgst > 0)
	state_component = sgst if sgst > 0 else utgst

	if has_igst and not has_state and cgst <= 0 and sgst <= 0 and utgst <= 0:
		return "igst"

	if has_igst and has_state:
		state_sum = cgst + state_component
		if (
			abs(igst - state_sum) <= max(2.0, state_sum * 0.05)
			and abs(cgst - state_component) <= max(2.0, max(cgst, state_component) * 0.10)
		):
			return "state"
		return "none"

	if (not has_igst) and has_state:
		if sgst > 0 and utgst > 0:
			return "none"
		if abs(cgst - state_component) <= max(2.0, max(cgst, state_component) * 0.10):
			return "state"

	return "none"


def _is_total_gst_brokerage_coherent(brok: float, gst_total: float, total: float) -> bool:
	if brok <= 0 or gst_total <= 0 or total <= 0:
		return False
	expected_total = brok + gst_total
	return abs(total - expected_total) <= max(2.0, expected_total * 0.02)


def _infer_tax_mode_fallback(row: Dict[str, str]) -> str:
	"""Infer a tax mode only when explicit components are absent but arithmetic/state clues are strong."""
	clear_mode = _detect_clear_tax_mode(row)
	if clear_mode != "none":
		return clear_mode

	brok = _amount_to_float(row.get("BROKERAGE Amount", ""))
	total = _amount_to_float(row.get("Total Inv Amt", ""))
	gst_total = _amount_to_float(row.get("GST TOTAL AMT", ""))
	cgst = _amount_to_float(row.get("CGST @ 9%", ""))
	sgst = _amount_to_float(row.get("SGST @ 9%", ""))
	utgst = _amount_to_float(row.get("UTGST", ""))
	igst = _amount_to_float(row.get("IGST", ""))

	if gst_total <= 0:
		return "none"

	has_tax_component = any(v > 0 for v in [cgst, sgst, utgst, igst])
	if has_tax_component:
		if igst > 0 and cgst <= 0 and sgst <= 0 and utgst <= 0:
			return "igst"
		if igst <= 0 and cgst > 0 and (sgst > 0 or utgst > 0):
			return "state"
		return "none"

	if not _is_total_gst_brokerage_coherent(brok, gst_total, total):
		return "none"

	balic_state = normalize_state_name((row.get("BALIC STATE", "") or "").strip())
	broker_state = normalize_state_name((row.get("BROKER GSTN STATE", "") or "").strip())

	if not balic_state:
		balic_state = gstin_to_state_name((row.get("BALIC GSTN", "") or "").strip().upper())
	if not broker_state:
		broker_state = gstin_to_state_name((row.get("BROKER GSTN", "") or "").strip().upper())

	if balic_state and broker_state:
		return "state" if balic_state.lower() == broker_state.lower() else "igst"

	return "none"


def _maybe_correct_brokerage_from_total_and_tax(row: Dict[str, str]) -> Dict[str, str]:
	"""Correct brokerage only when total and tax strongly imply a valid 18% GST relationship."""
	result = dict(row)
	total = _amount_to_float(result.get("Total Inv Amt", ""))
	brok = _amount_to_float(result.get("BROKERAGE Amount", ""))
	gst_total = _amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = _amount_to_float(result.get("CGST @ 9%", ""))
	sgst = _amount_to_float(result.get("SGST @ 9%", ""))
	utgst = _amount_to_float(result.get("UTGST", ""))
	igst = _amount_to_float(result.get("IGST", ""))

	state_component = sgst if sgst > 0 else utgst
	tax_component = igst if igst > 0 else (cgst + state_component)
	if tax_component <= 0 and gst_total > 0:
		tax_component = gst_total
	elif tax_component > 0 and gst_total > 0 and abs(gst_total - tax_component) <= max(2.0, tax_component * 0.05):
		tax_component = gst_total

	if total <= 0 or tax_component <= 0:
		return result

	implied_brok = total - tax_component
	if implied_brok <= 0:
		return result

	tax_ratio = tax_component / implied_brok
	if tax_ratio < 0.16 or tax_ratio > 0.20:
		return result

	if brok > 0 and abs(brok - implied_brok) <= max(2.0, brok * 0.02):
		return result

	if brok > 0:
		current_total = brok + tax_component
		if abs(total - current_total) <= max(2.0, current_total * 0.02):
			return result

	result["BROKERAGE Amount"] = _fmt_amount(implied_brok)
	return result


def _maybe_fix_ujjivan_gst_total(row: Dict[str, str]) -> Dict[str, str]:
	"""Fix Ujjivan rows where GST TOTAL is mis-mapped but tax components + total are coherent."""
	result = dict(row)
	source_low = (result.get("Source File", "") or "").strip().lower()
	if "ujjivan" not in source_low:
		return result

	brok = _amount_to_float(result.get("BROKERAGE Amount", ""))
	total = _amount_to_float(result.get("Total Inv Amt", ""))
	gst_total = _amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = _amount_to_float(result.get("CGST @ 9%", ""))
	sgst = _amount_to_float(result.get("SGST @ 9%", ""))
	utgst = _amount_to_float(result.get("UTGST", ""))
	igst = _amount_to_float(result.get("IGST", ""))

	state_component = sgst if sgst > 0 else utgst
	tax_component = igst if igst > 0 else (cgst + state_component)
	if brok <= 0 or total <= 0 or gst_total <= 0 or tax_component <= 0:
		return result

	# Require coherent total with extracted tax components.
	expected_total = brok + tax_component
	if abs(total - expected_total) > max(2.0, expected_total * 0.02):
		return result

	# Only override when GST TOTAL is clearly inconsistent with component tax.
	if abs(gst_total - tax_component) <= max(2.0, tax_component * 0.02):
		return result

	result["GST TOTAL AMT"] = _fmt_amount(tax_component)
	return result


def _maybe_fix_city_union_gst_total(row: Dict[str, str]) -> Dict[str, str]:
	"""Fix City Union rows where GST TOTAL is captured as roughly half of state-tax sum."""
	result = dict(row)
	source_low = (result.get("Source File", "") or "").strip().lower()
	if "city union bank" not in source_low:
		return result

	brok = _amount_to_float(result.get("BROKERAGE Amount", ""))
	total = _amount_to_float(result.get("Total Inv Amt", ""))
	gst_total = _amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = _amount_to_float(result.get("CGST @ 9%", ""))
	sgst = _amount_to_float(result.get("SGST @ 9%", ""))
	utgst = _amount_to_float(result.get("UTGST", ""))
	igst = _amount_to_float(result.get("IGST", ""))

	state_component = sgst if sgst > 0 else utgst
	state_sum = cgst + state_component
	if igst > 0 or brok <= 0 or total <= 0 or gst_total <= 0 or cgst <= 0 or state_component <= 0:
		return result
	if abs(cgst - state_component) > max(2.0, max(cgst, state_component) * 0.05):
		return result

	if abs(total - (brok + state_sum)) > max(2.0, total * 0.01):
		return result

	# Typical City Union issue: GST TOTAL is about half of CGST+SGST sum.
	ratio = gst_total / state_sum if state_sum > 0 else 0.0
	if ratio < 0.35 or ratio > 0.65:
		return result

	result["GST TOTAL AMT"] = _fmt_amount(state_sum)
	return result


def _maybe_fix_dhanlaxmi_tax_components(row: Dict[str, str]) -> Dict[str, str]:
	"""Repair Dhanlaxmi rows where one state-tax component is missing/corrupted but GST total is coherent."""
	result = dict(row)
	source_low = (result.get("Source File", "") or "").strip().lower()
	if "dhanlaxmi bank" not in source_low:
		return result

	brok = _amount_to_float(result.get("BROKERAGE Amount", ""))
	total = _amount_to_float(result.get("Total Inv Amt", ""))
	gst_total = _amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = _amount_to_float(result.get("CGST @ 9%", ""))
	sgst = _amount_to_float(result.get("SGST @ 9%", ""))
	utgst = _amount_to_float(result.get("UTGST", ""))
	igst = _amount_to_float(result.get("IGST", ""))

	if igst > 0 or gst_total <= 0:
		return result
	if brok > 0 and total > 0:
		tax_from_total = total - brok
		if tax_from_total <= 0 or abs(tax_from_total - gst_total) > max(2.0, gst_total * 0.02):
			return result

	tol = max(2.0, gst_total * 0.02)
	half = gst_total / 2.0
	state_component = sgst if sgst > 0 else utgst

	# SGST-only/UTGST-only rows where GST TOTAL already equals 2 * component.
	if cgst <= 0 and state_component > 0 and abs(gst_total - (2.0 * state_component)) <= tol:
		cgst = state_component
	# CGST-only rows where GST TOTAL already equals 2 * component.
	elif state_component <= 0 and cgst > 0 and abs(gst_total - (2.0 * cgst)) <= tol:
		sgst = cgst
		utgst = 0.0
	# Unbalanced two-component rows: align both to half of GST TOTAL.
	elif cgst > 0 and state_component > 0 and abs(cgst - state_component) > max(2.0, max(cgst, state_component) * 0.10):
		close_to_half = (abs(cgst - half) <= tol) or (abs(state_component - half) <= tol)
		if close_to_half:
			cgst = half
			sgst = half
			utgst = 0.0

	if cgst > 0:
		result["CGST @ 9%"] = _fmt_amount(cgst)
	if sgst > 0:
		result["SGST @ 9%"] = _fmt_amount(sgst)
		result["UTGST"] = ""
	elif utgst > 0:
		result["UTGST"] = _fmt_amount(utgst)
		result["SGST @ 9%"] = ""

	return result


def apply_confident_math_fill(row: Dict[str, str]) -> Dict[str, str]:
	"""
	Fill GST/total fields from arithmetic only when both are true:
	1) Brokerage amount is strongly validated against 18% tax math.
	2) Tax mode is clearly extracted as IGST-only or CGST+SGST/UTGST.
	"""
	result = dict(row)
	brok = _amount_to_float(result.get("BROKERAGE Amount", ""))
	if brok <= 0:
		return result

	mode = _detect_clear_tax_mode(result)
	if mode == "none":
		return result

	expected_gst = brok * 0.18
	expected_total = brok + expected_gst
	tol_gst = max(2.0, expected_gst * 0.02)
	tol_total = max(2.0, expected_total * 0.02)

	total = _amount_to_float(result.get("Total Inv Amt", ""))
	gst_total = _amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = _amount_to_float(result.get("CGST @ 9%", ""))
	sgst = _amount_to_float(result.get("SGST @ 9%", ""))
	utgst = _amount_to_float(result.get("UTGST", ""))
	igst = _amount_to_float(result.get("IGST", ""))

	brokerage_confident = False
	if total > 0 and abs(total - expected_total) <= tol_total:
		brokerage_confident = True
	if gst_total > 0 and abs(gst_total - expected_gst) <= tol_gst:
		brokerage_confident = True

	if mode == "igst" and igst > 0 and abs(igst - expected_gst) <= tol_gst:
		brokerage_confident = True
	if mode == "state":
		state_component = sgst if sgst > 0 else utgst
		half = expected_gst / 2.0
		tol_half = max(2.0, half * 0.02)
		if cgst > 0 and state_component > 0:
			if abs(cgst - half) <= tol_half and abs(state_component - half) <= tol_half:
				brokerage_confident = True

	if not brokerage_confident:
		return result

	if mode == "igst":
		result["CGST @ 9%"] = ""
		result["SGST @ 9%"] = ""
		result["UTGST"] = ""
		if igst <= 0:
			result["IGST"] = _fmt_amount(expected_gst)
			igst = expected_gst
		if gst_total <= 0:
			result["GST TOTAL AMT"] = _fmt_amount(igst if igst > 0 else expected_gst)

	elif mode == "state":
		half = expected_gst / 2.0
		result["IGST"] = ""
		if cgst <= 0:
			result["CGST @ 9%"] = _fmt_amount(half)
		if sgst <= 0 and utgst <= 0:
			result["SGST @ 9%"] = _fmt_amount(half)
			result["UTGST"] = ""
		elif sgst > 0 and utgst <= 0:
			result["SGST @ 9%"] = _fmt_amount(sgst)
			result["UTGST"] = ""
		elif utgst > 0 and sgst <= 0:
			result["UTGST"] = _fmt_amount(utgst)
			result["SGST @ 9%"] = ""

		cgst_f = _amount_to_float(result.get("CGST @ 9%", ""))
		state_component_f = _amount_to_float(result.get("SGST @ 9%", ""))
		if state_component_f <= 0:
			state_component_f = _amount_to_float(result.get("UTGST", ""))
		state_sum = cgst_f + state_component_f
		if gst_total <= 0 and state_sum > 0:
			result["GST TOTAL AMT"] = _fmt_amount(state_sum)

	gst_after_fill = _amount_to_float(result.get("GST TOTAL AMT", ""))
	if total <= 0 and gst_after_fill > 0:
		result["Total Inv Amt"] = _fmt_amount(brok + gst_after_fill)

	return result


def apply_gst_autofill(row: Dict[str, str]) -> Dict[str, str]:
	"""Auto-fill GST components from Brokerage when brokerage and GST total strongly imply 18% tax.
	Only applies when brokerage is present and GST total matches ~18% of brokerage.
	If state-mode components exist, fills CGST and SGST as 9% each; otherwise fills IGST as 18%.
	"""
	result = dict(row)
	brok = _amount_to_float(result.get("BROKERAGE Amount", ""))
	if brok <= 0:
		return result
	gst_total = _amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = _amount_to_float(result.get("CGST @ 9%", ""))
	sgst = _amount_to_float(result.get("SGST @ 9%", ""))
	utgst = _amount_to_float(result.get("UTGST", ""))
	igst = _amount_to_float(result.get("IGST", ""))

	expected_gst = brok * 0.18
	tol = max(2.0, expected_gst * 0.05)

	# Require extracted GST total to be present and close to expected, otherwise refuse to overwrite.
	if gst_total <= 0 or abs(gst_total - expected_gst) > tol:
		return result

	# If math already valid, do nothing
	is_valid, _ = validate_math_extraction(result)
	if is_valid:
		return result

	# Decide mode: prefer state-mode if any state components are present
	has_state = (cgst > 0 or sgst > 0 or utgst > 0)

	if has_state:
		# Fill CGST and SGST/UTGST as 9% each of brokerage
		half = brok * 0.09
		result["CGST @ 9%"] = _fmt_amount(half)
		# prefer SGST over UTGST if either present originally
		if sgst > 0 or (not sgst and not utgst):
			result["SGST @ 9%"] = _fmt_amount(half)
			result["UTGST"] = ""
		else:
			result["UTGST"] = _fmt_amount(half)
		result["IGST"] = ""
		result["GST TOTAL AMT"] = _fmt_amount(half * 2.0)
	else:
		# Fill IGST
		result["IGST"] = _fmt_amount(expected_gst)
		result["CGST @ 9%"] = ""
		result["SGST @ 9%"] = ""
		result["UTGST"] = ""
		result["GST TOTAL AMT"] = _fmt_amount(expected_gst)

	return result


def enforce_tax_mode_fields(row: Dict[str, str]) -> Dict[str, str]:
	"""Normalize row to a single tax mode by zeroing the non-selected mode."""
	clear_mode = _detect_clear_tax_mode(row)
	if clear_mode == "none":
		brok = _amount_to_float(row.get("BROKERAGE Amount", ""))
		balic_state = normalize_state_name((row.get("BALIC STATE", "") or "").strip())
		broker_state = normalize_state_name((row.get("BROKER GSTN STATE", "") or "").strip())
		if brok > 0 and balic_state and broker_state:
			clear_mode = "state" if balic_state.lower() == broker_state.lower() else "igst"
		else:
			clear_mode = _infer_tax_mode_fallback(row)
	if clear_mode == "none":
		return row

	brok = _amount_to_float(row.get("BROKERAGE Amount", ""))
	balic_state = normalize_state_name((row.get("BALIC STATE", "") or "").strip())
	broker_state = normalize_state_name((row.get("BROKER GSTN STATE", "") or "").strip())
	states_known = bool(balic_state and broker_state)
	states_match = states_known and balic_state.lower() == broker_state.lower()
	states_differ = states_known and balic_state.lower() != broker_state.lower()
	cgst = _amount_to_float(row.get("CGST @ 9%", ""))
	sgst = _amount_to_float(row.get("SGST @ 9%", ""))
	utgst = _amount_to_float(row.get("UTGST", ""))
	igst = _amount_to_float(row.get("IGST", ""))
	gst_total = _amount_to_float(row.get("GST TOTAL AMT", ""))
	state_component = max(sgst, utgst)
	state_tax = cgst + state_component

	mode = clear_mode
	if mode == "igst":
		row["CGST @ 9%"] = ""
		row["SGST @ 9%"] = ""
		row["UTGST"] = ""
		if igst <= 0 and brok > 0 and (states_differ or states_known):
			igst = brok * 0.18
		if igst <= 0 and gst_total > 0:
			igst = gst_total
		if igst > 0:
			row["IGST"] = _fmt_amount(igst)
		else:
			row["IGST"] = ""
		if gst_total <= 0 and igst > 0:
			row["GST TOTAL AMT"] = _fmt_amount(igst)
	elif mode == "state":
		row["IGST"] = ""
		if cgst <= 0 and brok > 0 and states_match:
			cgst = brok * 0.09
		if sgst <= 0 and utgst <= 0 and brok > 0 and states_match:
			sgst = brok * 0.09
		if sgst <= 0 and utgst > 0:
			sgst = utgst
			utgst = 0.0
		if cgst <= 0 and sgst <= 0 and gst_total > 0:
			half = gst_total / 2.0
			cgst = half
			sgst = half
			utgst = 0.0
		if cgst > 0:
			row["CGST @ 9%"] = _fmt_amount(cgst)
		else:
			row["CGST @ 9%"] = ""
		if sgst > 0:
			row["SGST @ 9%"] = _fmt_amount(sgst)
		else:
			row["SGST @ 9%"] = ""
		row["UTGST"] = ""
		state_total = cgst + max(sgst, utgst)
		if gst_total <= 0 and state_total > 0:
			row["GST TOTAL AMT"] = _fmt_amount(state_total)
	return row


def should_prioritize_ai_first(source_path: Optional[Path], source_display: str, page_text: str) -> bool:
	low = f"{str(source_path or '')} {source_display} {page_text[:1200]}".lower()
	triggers = [
		"city union",
		"city union bank",
		"probitas",
		"probitas insurance brokers",
		"turtlemint",
		"trusttech",
		"capri",
		"catalyst",
		"axis",
		"dbs",
		"motilal",
	]
	return any(token in low for token in triggers)


def _looks_mapping_incomplete(row: Dict[str, str]) -> bool:
	# Trigger LLM rescue when core identity/header fields are missing.
	if not (row.get("Agent PAN", "") or "").strip():
		return True
	if not (row.get("Name of Service Receipient", "") or "").strip():
		return True
	if not (row.get("BROKER GSTN", "") or "").strip():
		return True
	if not (row.get("BALIC STATE", "") or "").strip():
		return True
	if not (row.get("Vendor Inv Date", "") or "").strip():
		return True
	if not (row.get("Vendor Inv No", "") or "").strip():
		return True
	if not (row.get("Total Inv Amt", "") or "").strip():
		return True
	if not (row.get("GST TOTAL AMT", "") or "").strip():
		return True
	if not (row.get("Narration", "") or "").strip():
		return True
	return False


def _has_gstin_role_conflict(row: Dict[str, str]) -> bool:
	"""Detect BALIC/BROKER GSTIN role collisions and obvious broker mis-maps."""
	balic_gstn = (row.get("BALIC GSTN", "") or "").strip().upper()
	broker_gstn = (row.get("BROKER GSTN", "") or "").strip().upper()
	if balic_gstn and broker_gstn and balic_gstn == broker_gstn:
		return True

	recipient_low = (row.get("Name of Service Receipient", "") or "").strip().lower()
	agent_pan = (row.get("Agent PAN", "") or "").strip().upper()
	if broker_gstn and ("bajaj" in recipient_low or "allianz" in recipient_low):
		# Broker GSTIN should not be a BALIC-family GSTIN when recipient is BALIC.
		if "AADCA1701E" in broker_gstn:
			return True
		if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", agent_pan) and agent_pan not in broker_gstn:
			return True

	return False


def _looks_mapping_improbable(row: Dict[str, str]) -> bool:
	"""Detect suspicious numeric mapping where LLM rescue is useful."""
	normalized = enforce_tax_mode_fields(dict(row))

	if _has_gstin_role_conflict(normalized):
		return True

	is_math_valid, _ = validate_math_extraction(normalized)
	if not is_math_valid:
		return True

	try:
		brok = float((normalized.get("BROKERAGE Amount", "") or "").replace(",", "")) if (normalized.get("BROKERAGE Amount", "") or "").strip() else 0.0
		total = float((normalized.get("Total Inv Amt", "") or "").replace(",", "")) if (normalized.get("Total Inv Amt", "") or "").strip() else 0.0
		cgst = float((normalized.get("CGST @ 9%", "") or "").replace(",", "")) if (normalized.get("CGST @ 9%", "") or "").strip() else 0.0
		sgst = float((normalized.get("SGST @ 9%", "") or "").replace(",", "")) if (normalized.get("SGST @ 9%", "") or "").strip() else 0.0
		utgst = float((normalized.get("UTGST", "") or "").replace(",", "")) if (normalized.get("UTGST", "") or "").strip() else 0.0
		igst = float((normalized.get("IGST", "") or "").replace(",", "")) if (normalized.get("IGST", "") or "").strip() else 0.0
		gst_total = float((normalized.get("GST TOTAL AMT", "") or "").replace(",", "")) if (normalized.get("GST TOTAL AMT", "") or "").strip() else 0.0

		state_tax = max(sgst, utgst)

		# Mutually exclusive tax modes should not coexist strongly.
		if igst > 0 and (cgst > 0 or state_tax > 0):
			return True

		# Total should not be lower than brokerage in valid mapped receipts.
		if brok > 0 and total > 0 and total < brok:
			return True

		# GST total should align with either IGST or state-tax sum.
		if gst_total > 0:
			expected_gst = igst if igst > 0 else (cgst + state_tax)
			if expected_gst > 0 and abs(gst_total - expected_gst) > max(2.0, expected_gst * 0.05):
				return True

		# Tax ratio guard catches obvious mis-maps (e.g., IDs/HSN values in amount columns).
		if brok > 0:
			tax_component = igst if igst > 0 else (cgst + state_tax)
			if tax_component > 0:
				ratio = tax_component / brok
				if ratio < 0.05 or ratio > 0.30:
					return True
			if gst_total > 0:
				gst_ratio = gst_total / brok
				if gst_ratio < 0.05 or gst_ratio > 0.30:
					return True

		# HSN/SAC-like leakage often appears as very large integer in GST amount fields.
		if gst_total >= 900000:
			return True
	except (ValueError, TypeError, AttributeError, ZeroDivisionError):
		return True

	return False


def _looks_gst_mode_ambiguous(row: Dict[str, str]) -> bool:
	"""Detect rows where tax values exist but GST mode is not clearly resolvable."""
	total = _amount_to_float(row.get("Total Inv Amt", ""))
	brok = _amount_to_float(row.get("BROKERAGE Amount", ""))
	gst_total = _amount_to_float(row.get("GST TOTAL AMT", ""))
	cgst = _amount_to_float(row.get("CGST @ 9%", ""))
	sgst = _amount_to_float(row.get("SGST @ 9%", ""))
	utgst = _amount_to_float(row.get("UTGST", ""))
	igst = _amount_to_float(row.get("IGST", ""))
	mode = _detect_clear_tax_mode(row)
	has_tax_bits = any(v > 0 for v in [cgst, sgst, utgst, igst])
	if gst_total <= 0:
		return False
	if mode == "none":
		return True
	if mode in {"igst", "state"} and has_tax_bits:
		# Still ambiguous when GST TOTAL exists but the row also carries partial or mixed tax cues.
		if mode == "igst" and (cgst > 0 or sgst > 0 or utgst > 0):
			return True
		if mode == "state" and igst > 0:
			return True
		if brok > 0 and total > 0:
			expected_total = brok + gst_total
			if abs(total - expected_total) > max(2.0, expected_total * 0.02):
				return True
		return False
	if brok > 0 and total > 0:
		return True
	return False


def _gst_mode_context_hint(rows: List[Dict[str, str]]) -> str:
	"""Summarize ambiguous GST rows for LLM fallback."""
	parts: List[str] = []
	for idx, row in enumerate(rows, start=1):
		brok = _amount_to_float(row.get("BROKERAGE Amount", ""))
		total = _amount_to_float(row.get("Total Inv Amt", ""))
		gst_total = _amount_to_float(row.get("GST TOTAL AMT", ""))
		cgst = _amount_to_float(row.get("CGST @ 9%", ""))
		sgst = _amount_to_float(row.get("SGST @ 9%", ""))
		utgst = _amount_to_float(row.get("UTGST", ""))
		igst = _amount_to_float(row.get("IGST", ""))
		mode = _detect_clear_tax_mode(row)
		parts.append(
			f"row{idx}: Brokerage={_fmt_amount(brok) if brok > 0 else 'blank'}, "
			f"Total={_fmt_amount(total) if total > 0 else 'blank'}, "
			f"GST TOTAL={_fmt_amount(gst_total) if gst_total > 0 else 'blank'}, "
			f"CGST={_fmt_amount(cgst) if cgst > 0 else '0'}, SGST={_fmt_amount(sgst) if sgst > 0 else '0'}, "
			f"UTGST={_fmt_amount(utgst) if utgst > 0 else '0'}, IGST={_fmt_amount(igst) if igst > 0 else '0'}, mode={mode}"
		)
	return " | ".join(parts)


def extract_receipts_with_azure_llm(page_text: str, source_file: str, page_num: int, context_hint: str = "") -> List[Dict[str, str]]:
	global AZURE_OPENAI_WORKING_DEPLOYMENT
	global AZURE_AI_CALL_COUNT
	global AZURE_AI_INPUT_CHARS
	global AZURE_AI_OUTPUT_CHARS
	config = get_azure_openai_config()
	if not config:
		return []
	if len((page_text or "").strip()) < 40:
		return []

	endpoint, api_key, deployment, api_version = config
	system_context_hint = (
		"You are extracting commission invoice mappings from OCR text. "
		"Your job is to map fields exactly, not to summarize. "
		"Return JSON only with top-level key receipts. "
		"If a page has no actual receipt, return {\"receipts\": []}. "
		"Never invent missing values. Never copy HSN/SAC codes into GST fields. "
		"Prefer leaving a field blank over guessing. "
	)
	bajaj_finance_context = _is_bajaj_finance_context(f"{source_file} {page_text}")

	system_prompt = (
		f"{system_context_hint} "
		"A single page can contain one receipt, multiple receipts, or no receipt. "
		"Use keys exactly: Agent Name, Agent PAN, Name of Service Receipient, BALIC STATE, BALIC GSTN, BROKER GSTN STATE, BROKER GSTN, Vendor Inv Date, Vendor Inv No, Total Inv Amt, BROKERAGE Amount, CGST @ 9%, SGST @ 9%, UTGST, IGST, GST TOTAL AMT, Narration. "
		"Field rules: "
		"1) BALIC GSTN is the recipient/company GSTIN. If a GSTIN belongs to Bajaj Allianz/BALIC, place it in BALIC GSTN. "
		"2) BROKER GSTN is the broker/agent GSTIN; never duplicate BALIC GSTN into BROKER GSTN unless the text explicitly says the same GSTIN is used for both roles. "
		"3) Vendor Inv No must be the invoice identifier only, not GSTIN, PAN, HSN, SAC, dates, page numbers, totals, or references. "
		"4) Vendor Inv Date must be a date. If no date is present, leave blank. "
		"5) Total Inv Amt is the invoice total. BROKERAGE Amount is the taxable brokerage amount. GST TOTAL AMT is only the tax amount or tax total. "
		"6) Use either IGST or CGST+SGST/UTGST, not both. If the page clearly shows interstate tax, keep IGST and blank the state tax fields. If it clearly shows state tax, keep CGST/SGST or UTGST and blank IGST. If tax mode is confusing, inspect which tax cells are non-zero and map them exactly: zero fields stay blank, non-zero fields stay populated, and GST TOTAL should match the non-zero tax mode. Do not mix modes. "
		"7) Narration should be a short business label such as COMMISSION, and can be blank if truly absent. "
		"8) If OCR text contains multiple receipts on one page, return one object per receipt. "
		"9) Do not fabricate invoice numbers from file names unless the invoice number is explicitly visible in OCR. "
		"10) If numbers disagree or the mapping is uncertain, leave the field blank rather than guessing."
	)
	if bajaj_finance_context:
		system_prompt += (
			" Bajaj Finance specific rule: return only the single summary row whose Particulars cell says Total. "
			"Ignore intermediate line items and do not split each line item into a separate receipt. "
			"If no Total row is visible, return an empty receipts array."
		)
	user_prompt = (
		f"Source file: {source_file}\n"
		f"Page: {page_num}\n"
		"Extract receipts from this text. Preserve only values that are explicitly supported by the OCR. "
		"If a value is ambiguous, blank it. Use the same field names exactly. "
		f"{('Structured GST context: ' + context_hint + chr(10)) if context_hint else ''}"
		"Text:\n"
		f"{page_text[:24000]}"
	)

	payload = {
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		"temperature": 0,
		"max_tokens": 1800,
		"response_format": {"type": "json_object"},
	}

	deployments = get_azure_openai_deployment_candidates(deployment)
	if AZURE_OPENAI_WORKING_DEPLOYMENT and AZURE_OPENAI_WORKING_DEPLOYMENT in deployments:
		deployments = [AZURE_OPENAI_WORKING_DEPLOYMENT] + [d for d in deployments if d != AZURE_OPENAI_WORKING_DEPLOYMENT]

	for dep in deployments:
		url = (
			f"{endpoint}/openai/deployments/{urllib.parse.quote(dep, safe='')}/chat/completions"
			f"?api-version={urllib.parse.quote(api_version, safe='')}"
		)
		try:
			AZURE_AI_CALL_COUNT += 1
			AZURE_AI_INPUT_CHARS += len(page_text[:24000])
			request = urllib.request.Request(
				url=url,
				data=json.dumps(payload).encode("utf-8"),
				headers={
					"Content-Type": "application/json",
					"api-key": api_key,
				},
				method="POST",
			)
			with urllib.request.urlopen(request, timeout=45) as response:
				body = response.read().decode("utf-8", errors="ignore")
			parsed = json.loads(body)
			content = (
				parsed.get("choices", [{}])[0]
				.get("message", {})
				.get("content", "")
			)
			AZURE_AI_OUTPUT_CHARS += len(content or "")
			if not content:
				AZURE_OPENAI_WORKING_DEPLOYMENT = dep
				return []
			content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
			content = re.sub(r"\s*```$", "", content.strip())
			obj = json.loads(content)
			receipts = obj.get("receipts", []) if isinstance(obj, dict) else []
			if not isinstance(receipts, list):
				AZURE_OPENAI_WORKING_DEPLOYMENT = dep
				return []
			rows: List[Dict[str, str]] = []
			for item in receipts:
				if not isinstance(item, dict):
					continue
				row = _ai_row_to_output_fields(item)
				if is_valid_invoice_no(row.get("Vendor Inv No", "")):
					rows.append(row)
			AZURE_OPENAI_WORKING_DEPLOYMENT = dep
			return rows
		except urllib.error.HTTPError as exc:
			if exc.code == 404:
				continue
			LOGGER.warning("Azure receipt extraction failed on %s page %s (deployment=%s): %s", source_file, page_num, dep, exc)
			return []
		except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
			LOGGER.warning("Azure receipt extraction failed on %s page %s (deployment=%s): %s", source_file, page_num, dep, exc)
			return []

	LOGGER.warning("Azure receipt extraction failed on %s page %s: no valid deployment found", source_file, page_num)
	return []


def _ocr_cache_enabled() -> bool:
	# Enabled by default. Set OCR_CACHE=0/off/false to disable.
	raw = (os.getenv("OCR_CACHE") or "1").strip().lower()
	return raw not in {"0", "off", "false", "no"}


def _ocr_cache_key(source_id: str, pdf_digest: str, page_num: int, prefer_google: bool, force_google: bool) -> str:
	seed = f"{source_id}|{pdf_digest}|p={page_num}|pg={int(prefer_google)}|fg={int(force_google)}"
	return hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()


def _ocr_cache_get(cache_key: str) -> str:
	if not _ocr_cache_enabled():
		return ""
	cache_file = OCR_CACHE_DIR / f"{cache_key}.txt"
	if cache_file.exists():
		try:
			return normalize_text(cache_file.read_text(encoding="utf-8", errors="ignore"))
		except Exception:
			return ""
	return ""


def _ocr_cache_put(cache_key: str, text: str) -> None:
	if not _ocr_cache_enabled():
		return
	try:
		OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
		cache_file = OCR_CACHE_DIR / f"{cache_key}.txt"
		cache_file.write_text(text or "", encoding="utf-8")
	except Exception:
		pass


def clean_amount(value: str) -> str:
	value = value.strip()
	# Handle OCR numbers with decimal comma, e.g. 228524,00 -> 228524.00
	if "," in value and "." not in value and re.search(r",\d{1,2}\b", value):
		value = re.sub(r"[^0-9,\-]", "", value)
		parts = value.split(",")
		if len(parts) >= 2 and parts[-1].isdigit() and 1 <= len(parts[-1]) <= 2:
			integer_part = "".join(parts[:-1])
			decimal_part = parts[-1].ljust(2, "0")
			value = f"{integer_part}.{decimal_part}"
		else:
			value = value.replace(",", "")
	else:
		value = value.replace(",", "")
	match = re.search(r"-?\d+(?:\.\d{1,2})?", value)
	return match.group(0) if match else ""

def validate_and_correct_taxes(
	taxable_amount: str, cgst: str, sgst: str, utgst: str, igst: str
) -> Tuple[str, str, str, str, str]:
	"""
	Sanity check tax amounts without mutating extracted values.
	Expected rates: CGST @ 9%, SGST @ 9%, IGST @ 18%.
	Returns original values so downstream validation can flag suspicious rows.
	"""
	return cgst, sgst, utgst, igst



INDIAN_STATES = [
	"Andhra Pradesh",
	"ARUNACHAL PRADESH",
	"Assam",
	"Bihar",
	"Chattisgarh",
	"Goa",
	"Gujarat",
	"Haryana",
	"Himachal Pradesh",
	"Jharkhand",
	"Karnataka",
	"Kerala",
	"Madhya Pradesh",
	"Maharashtra",
	"Manipur",
	"Meghalaya",
	"Mizoram",
	"Nagaland",
	"Odisha",
	"Punjab",
	"Rajasthan",
	"Sikkim",
	"Tamil Nadu",
	"Telangana",
	"Tripura",
	"Uttar Pradesh",
	"Uttarakhand",
	"Bengal",
	"Andaman and Nicobar",
	"Chandigarh",
	"DAMAN AND DIU",
	"DADRA AND NAGAR HAVELI",
	"Delhi",
	"Jammu And Kashmir",
	"Ladakh",
	"Lakshadweep",
	"PUDUCHERRY",
]


def normalize_state_name(raw_state: str) -> str:
	if not raw_state:
		return ""
	candidate = re.sub(r"\s+", " ", raw_state).strip()
	if not candidate:
		return ""
	alias_candidate = STATE_NAME_ALIASES.get(candidate.lower(), "")
	if alias_candidate:
		return alias_candidate

	for state in INDIAN_STATES:
		if candidate.lower() == state.lower():
			return state

	letters_only = re.sub(r"[^a-z]", "", candidate.lower())
	if not letters_only:
		return ""

	state_map = {re.sub(r"[^a-z]", "", s.lower()): s for s in INDIAN_STATES}
	close = difflib.get_close_matches(letters_only, list(state_map.keys()), n=1, cutoff=0.72)
	if close:
		return state_map[close[0]]

	for key, state in state_map.items():
		if letters_only in key or key in letters_only:
			return state

	return ""


def gstin_to_pan(value: str) -> str:
	candidate = (value or "").strip().upper()
	if not is_valid_gstin(candidate):
		return ""
	return candidate[2:12]


def normalize_balic_company_name(raw_name: str) -> str:
	if not raw_name:
		return ""
	name = re.sub(r"\s+", " ", raw_name).strip()
	name_low = name.lower()
	if name_low in {"namebaddress", "name/address", "name and address", "name & address"}:
		return ""
	if name_low in {"insurance limited", "insurance company ltd", "insurance company limited"}:
		return ""
	if "bajaj life insurance limited" in name_low and "allianz" not in name_low:
		return "Bajaj Life Insurance Limited"
	if (
		"leela chambers" in name_low
		or "bala lte irsurance umited" in name_low
		or "bala lte insurance" in name_low
		or ("satara road" in name_low and "aranyeshwar" in name_low)
	):
		return "Bajaj Allianz Life Insurance Company Ltd"
	letters_only = re.sub(r"[^a-z]", "", name_low)
	canon_letters = re.sub(r"[^a-z]", "", "bajaj allianz life insurance company ltd")

	if "bajaj" in letters_only:
		close = difflib.get_close_matches(letters_only, [canon_letters], n=1, cutoff=0.48)
		if close:
			return "Bajaj Allianz Life Insurance Company Ltd"

	# OCR-friendly fallback when Bajaj is misspelled but Allianz/Life/Insurance signals are present.
	if (("allianz" in name_low) or ("aianz" in name_low) or ("amlanz" in name_low) or ("alianz" in name_low)) and (
		("life" in name_low) or ("insurance" in name_low)
	):
		return "Bajaj Allianz Life Insurance Company Ltd"

	if "bajaj" in name_low and (("allianz" in name_low) or ("aianz" in name_low) or ("amlanz" in name_low) or ("alianz" in name_low)):
		return "Bajaj Allianz Life Insurance Company Ltd"

	if "bajaj" in name_low and "life" in name_low and "insurance" in name_low:
		return "Bajaj Allianz Life Insurance Company Ltd"

	return name


def is_known_balic_entity(name: str) -> bool:
	normalized = normalize_balic_company_name(sanitize_party_name(name or ""))
	return normalized in {
		"Bajaj Allianz Life Insurance Company Ltd",
		"Bajaj Life Insurance Limited",
	}


def normalize_corporate_agent_name(raw_name: str) -> str:
	if not raw_name:
		return ""
	name = _strip_trailing_context_noise(raw_name)
	low = name.lower()
	if "bajaj housing" in low:
		return "Bajaj Housing Finance Limited"
	if "bajaj auto finance" in low or "bajaj finance" in low:
		return "Bajaj Finance Limited"
	# explicit override: treat 'Bajaj Auto' mentions as Bajaj Housing
	if 'bajaj auto' in low:
		return 'Bajaj Housing Finance Limited'
	if "finozone" in low:
		return "Finozone Financial Services"
	if "jammu" in low and "kashmir" in low and "bank" in low:
		return "Jammu & Kashmir Bank"
	if "nkgsb" in low:
		return "NKGSB Bank"
	if "axis" in low and "bank" in low:
		return "Axis Bank Ltd"
	if "dbs" in low and "bank" in low:
		return "DBS Bank India Limited"
	return name


def sanitize_party_name(raw_name: str) -> str:
	if not raw_name:
		return ""
	name = _strip_trailing_context_noise(raw_name)
	name = re.sub(r"^(?:of\s+)+", "", name, flags=re.IGNORECASE)
	name = re.sub(r"^(?:billed\s*to|bill\s*to|ship\s*to|service\s*recipient)\s*[:\-]?\s*", "", name, flags=re.IGNORECASE)
	low = name.lower()
	if len(name) < 6:
		return ""
	if low in {"insurance limited", "limited", "ltd", "of delivery", "delivery", "details", "billed to", "bill to", "ship to"}:
		return ""
	if low.startswith("details"):
		return ""
	if low.startswith("billed to") or low.startswith("bill to") or low.startswith("ship to"):
		return ""
	if any(token in low for token in ["state code", "gstin", "invoice no", "invoice date", "particulars"]):
		return ""
	if any(token in low for token in ["road", "street", "nagar", "layout", "floor", "building", "near", "sector", "colony", "chennai", "mumbai", "bengaluru", "pincode", "pin code"]):
		return ""
	if len(re.findall(r"\d", name)) >= 4:
		return ""
	return name


def extract_company_candidates(text: str) -> List[str]:
	pattern = r"\b([A-Z][A-Za-z&.,()'/-]{2,90}?(?:Limited|Ltd|Bank(?:\s+India)?(?:\s+Limited)?))\b"
	candidates = []
	seen = set()
	for match in re.finditer(pattern, text):
		candidate = sanitize_party_name(match.group(1))
		if not candidate:
			continue
		key = candidate.lower()
		if key in seen:
			continue
		seen.add(key)
		candidates.append(candidate)
	return candidates


def extract_provider_receiver_names(text: str) -> Tuple[str, str]:
	flat = re.sub(r"\s+", " ", text)
	provider = ""
	receiver = ""
	flat_low = flat.lower()

	# Tax Invoice specific layout: provider/receiver values are often around C/o and Bajaj life insurance markers.
	if "tax invoice" in flat_low and "details of service" in flat_low:
		# For TAX invoices, look for explicit bank/institution names first
		bank_patterns = [
			r"(IDFC\s+(?:FIRST\s+)?BANK)",
			r"(Axis\s+Bank(?:\s+Ltd)?)",
			r"(HDFC\s+Bank(?:\s+Ltd)?)",
			r"(ICICI\s+Bank(?:\s+Ltd)?)",
			r"(SBI|State\s+Bank\s+of\s+India)",
			r"(Kotak\s+Mahindra\s+Bank)",
			r"(IndusInd\s+Bank)",
			r"(City\s+Union\s+Bank)",
			r"(Ujjivan\s+(?:Small\s+Finance\s+)?Bank)",
			r"(Federal\s+Bank)",
			r"(RBL\s+Bank)",
			r"(BOB|Bank\s+of\s+Baroda)",
			r"(Union\s+Bank)",
			r"(Canara\s+Bank)",
		]
		for pattern in bank_patterns:
			match = re.search(pattern, flat, re.IGNORECASE)
			if match:
				provider = sanitize_party_name(match.group(1))
				if provider:
					receiver = extract_bajaj_company_name(flat)
					return provider, receiver
		
		# Fallback for TAX invoices: extract from standard patterns
		tax_agent = find_first([
			r"c/o\s*([A-Za-z][A-Za-z\s&.,()'/-]{4,80}?(?:Limited|Ltd|Bank(?:\s+Ltd)?))",
			r"(?:service|goods)\s+provider\s*[:\-]?\s*([A-Za-z][A-Za-z\s&.,()'/-]{4,80}?(?:Limited|Ltd|Bank(?:\s+Ltd)?))",
		], flat)
		tax_receiver = find_first([
			r"(?:service|goods)\s+receiver\s*[:\-]?\s*([A-Za-z][A-Za-z\s&.,()'/-]{4,80}?(?:Limited|Ltd|Bank(?:\s+Ltd)?))",
		], flat)
		provider = sanitize_party_name(tax_agent)
		receiver = sanitize_party_name(tax_receiver)
		if not receiver:
			receiver = extract_bajaj_company_name(flat)

	provider_patterns = [
		r"details\s+of\s+supplier\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,120})",
		r"details\s+of\s+service(?:\s*/\s*goods)?\s+provider\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,120})",
		r"corporate\s+agent\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,120})",
	]
	receiver_patterns = [
		r"details\s+of\s+receiver\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,120})",
		r"details\s+of\s+recipient\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,120})",
		r"details\s+of\s+service(?:\s*/\s*goods)?\s+receiver\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,120})",
		r"details\s+of\s+service\s+recipient\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,120})",
	]

	if not provider:
		for pattern in provider_patterns:
			match = re.search(pattern, flat, re.IGNORECASE)
			if match:
				provider = sanitize_party_name(match.group(1))
				if provider:
					break

	if not receiver:
		for pattern in receiver_patterns:
			match = re.search(pattern, flat, re.IGNORECASE)
			if match:
				receiver = sanitize_party_name(match.group(1))
				if receiver:
					break

	if (not provider) or (not receiver):
		candidates = extract_company_candidates(text)
		if len(candidates) >= 2:
			if not provider:
				provider = candidates[0]
			if not receiver:
				# Prefer a Bajaj/Allianz/Housing/Insurance candidate for service recipient when available.
				preferred = ""
				for candidate in candidates[1:]:
					low = candidate.lower()
					if any(token in low for token in ["allianz", "bajaj", "insurance", "housing"]):
						preferred = candidate
						break
				receiver = preferred or candidates[1]

	return provider, receiver


def extract_agent_from_folder_fallback(source_path: Path) -> str:
	parts = str(source_path).replace("/", "\\").split("\\")
	for part in reversed(parts):
		if not part or part.lower().startswith("receipt_zip_"):
			continue
		base = re.sub(r"\.(pdf|jpg|jpeg|png|zip)$", "", part, flags=re.IGNORECASE)
		part_low = base.lower().strip()
		if any(skip in part_low for skip in ["extracted_invoices", "part 1", "part 2", "part3", "part4", "part5"]):
			continue
		if not any(key in part_low for key in ["insurance", "brok", "bank", "finance", "capital", "imf", "services", "ltd", "limited"]):
			continue
		cleaned = re.sub(r"[_-]+", " ", base)
		cleaned = re.sub(r"\b\d{1,2}\s*[-/]\s*\d{1,2}\b", " ", cleaned)
		cleaned = re.sub(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b\s*\d{2}", " ", cleaned, flags=re.IGNORECASE)
		cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_")
		return cleaned
	return ""


def extract_agent_from_folder_path(source_path: Path) -> str:
	"""Extract and normalize agent name from folder path."""
	path_str = str(source_path).lower()
	folder_parts = path_str.split("\\")
	
	# Search from most specific folder name backward
	for part in reversed(folder_parts):
		part_low = part.lower()
		
		# Banks (check specific patterns first to avoid partial matches)
		if "jammu" in part_low and "kashmir" in part_low:
			return "Jammu & Kashmir Bank"
		if "axis" in part_low and "outward" in part_low:
			return "Axis Bank Ltd"
		if "dbs" in part_low and "commission" in part_low:
			return "DBS Bank India Limited"
		if "nkgsb" in part_low:
			return "NKGSB Bank"
		if "axis bank" in part_low:
			return "Axis Bank Ltd"
		if "city union" in part_low:
			return "City Union Bank Ltd"
		if "dhanlaxmi" in part_low:
			return "Dhanlaxmi Bank Limited"
		if "idfc" in part_low:
			return "IDFC FIRST Bank"
		if "india post" in part_low:
			return "India Post Payments Bank Ltd"
		if "bajaj housing" in part_low:
			return "Bajaj Housing Finance Limited"
		if "bajaj auto finance" in part_low or "bajaj finance" in part_low:
			return "Bajaj Finance Limited"
		if "karur vysya" in part_low:
			return "Karur Vysya Bank Ltd"
		if "bajaj auto" in part_low:
			return "Bajaj Housing Finance Limited"
		if "finozone" in part_low:
			return "Finozone Financial Services"
		
		# Insurance brokers and fintech
		if "marsh india" in part_low:
			return "Marsh India Insurance Brokers Private Ltd"
		if "catalyst" in part_low:
			return "Catalyst Insurance Broking"
		if "claycove23" in part_low:
			return "Claycove23 Insurance Tech Private Limited"
		if "coverfox" in part_low:
			return "Coverfox Insurance Broking Pvt Ltd"
		if "ethika" in part_low:
			return "Ethika Insurance Broking Private Limited"
		if "finhaat" in part_low:
			return "Finhaat Insurance Broking Private Limited"
		if "finworkx" in part_low:
			return "Finworkx IMF"
		if "gennext" in part_low:
			return "Gennext Insurance Brokers Pvt Ltd"
		if "geojit" in part_low:
			return "Geojit Financial Services Ltd"
		if "gallagher" in part_low:
			return "Gallagher Insurance Brokers Pvt Ltd"
		if "ideal insurance" in part_low:
			return "Ideal Insurance Brokers Pvt Ltd"
		if "imf" in part_low and "multiple" not in part_low:
			return "IMF"
		if "incred financial" in part_low:
			return "Incred Financial Services Limited"
		if "india insure" in part_low:
			return "India Insure Risk Management"
		if "jb boda" in part_low:
			return "JB Boda Insurance and Reinsurance Brokers Pvt Ltd"
		if "km dastur" in part_low:
			return "KM Dastur Reinsurance Brokers Pvt Ltd"
		if "lifemart" in part_low:
			return "Lifemart Insurance Brokers Pvt Ltd"
		if "motilal oswal" in part_low:
			return "Motilal Oswal Financial Services Limited"
		if "muthoot" in part_low:
			return "Muthoot Insurance Brokers Pvt Ltd"
		if "aapt insurance" in part_low:
			return "Aapt Insurance Brokers Private Limited"
		if "aims insurance" in part_low:
			return "AIMS Insurance Broking Private Limited"
		if "alliance" in part_low and "insurance" in part_low:
			return "Alliance Insurance Brokers Pvt. Ltd."
		if "apeejay" in part_low:
			return "Apeejay Insurance Broking Services Private Limited"
		if "arham" in part_low:
			return "Arham Insurance Brokers Private Limited"
		if "beacon" in part_low:
			return "Beacon Insurance Brokers Pvt Ltd"
		if "capri global" in part_low:
			return "Capri Global Capital Limited"
		if "choice insurance" in part_low:
			return "Choice Insurance Broking India Pvt Ltd"
		if "coverkraft" in part_low:
			return "Coverkraft Insurance Brokers Private Limited"
		if "future first" in part_low:
			return "Future First Insurance Broking Private Limited"
		if "futurisk" in part_low:
			return "Futurisk Insurance Broking Co Pvt Ltd"
		if "harita insurance" in part_low:
			return "Harita Insurance Broking Llp"
		if "livlong" in part_low:
			return "Livlong Insurance Broker Limited"
		if "navinchandra" in part_low:
			return "Navinchandra Insurance Broking Private Limited"
		if "novo insurance" in part_low:
			return "Novo Insurance Broking Services Private Limited"
		if "probitas" in part_low:
			return "Probitas Insurance Brokers Pvt. Ltd"
		if "probus" in part_low:
			return "Probus Insurance Broker Private Limited"
		if "prudent" in part_low:
			return "Prudent Insurance Brokers Pvt Ltd"
		if "securenow" in part_low:
			return "Securenow Insurance Broker Pvt Ltd"
		if "sonnen" in part_low:
			return "Sonnen Insurance Broking Services Pvt Ltd"
		if "turtlemint" in part_low:
			return "Turtlemint Insurance Broking Services Pvt Ltd"
		if "ujjivan" in part_low:
			return "Ujjivan"
		if "unison insurance" in part_low:
			return "Unison Insurance Broking Services Pvt Ltd"
		if "xperitus" in part_low:
			return "Xperitus Insurance Brokers Pvt Ltd"
		if "yella" in part_low:
			return "Yella Insurance Broking Pvt Ltd"
		if "zoom insurance" in part_low:
			return "Zoom Insurance Brokers Pvt Ltd"
		if "mahindra" in part_low:
			return "Mahindra Insurance Brokers Pvt Ltd"
	
	return ""


def infer_corporate_agent_from_context(source_path: Path, text: str) -> str:
	context = f"{source_path} {text}"
	low = context.lower()

	# Prefer explicit agent mentions in text/context first (avoids blanket folder defaults like 'IMF').
	patterns = [
		r"corporate\s*agent\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,100}?)(?:\s+(?:gstin|pan|state|invoice|service|address|$))",
		r"agent\s*name\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,100}?)(?:\s+(?:gstin|pan|state|invoice|service|address|$))",
	]
	for pattern in patterns:
		match = re.search(pattern, context, re.IGNORECASE)
		if match:
			candidate = normalize_corporate_agent_name(match.group(1))
			if candidate and not _is_placeholder_agent_name(candidate):
				return candidate

	# Next prefer explicit provider extracted from document text
	provider_name, receiver_name = extract_provider_receiver_names(text)
	if provider_name and not _is_placeholder_agent_name(provider_name):
		return provider_name

	# Folder path fallback as last resort (previously caused IMF defaulting)
	folder_agent = extract_agent_from_folder_path(source_path)
	if folder_agent:
		return folder_agent

	fallback_agent = extract_agent_from_folder_fallback(source_path)
	if fallback_agent:
		return fallback_agent

	# Strong source-based routing
	if "finozone" in low:
		return "Finozone Financial Services"
	if "jammu and kashmir bank" in low or "j&k bank" in low or "j k bank" in low:
		return "Jammu & Kashmir Bank"
	if "nkgsb" in low:
		return "NKGSB Bank"
	if "invoices axis" in low or ("axis" in low and "outwardinvoice" in low):
		return "Axis Bank Ltd"
	if "july_25 dbs" in low or ("dbs" in low and "commission invoice" in low):
		return "DBS Bank India Limited"

	patterns = [
		r"corporate\s*agent\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,100}?)(?:\s+(?:gstin|pan|state|invoice|service|address|$))",
		r"agent\s*name\s*[:\-]?\s*([A-Za-z&.,()'/-]{4,100}?)(?:\s+(?:gstin|pan|state|invoice|service|address|$))",
	]
	for pattern in patterns:
		match = re.search(pattern, context, re.IGNORECASE)
		if not match:
			continue
		candidate = normalize_corporate_agent_name(match.group(1))
		if candidate and not _is_placeholder_agent_name(candidate):
			return candidate

	return ""


def _is_placeholder_agent_name(name: str) -> bool:
	"""Check if agent name is a placeholder or corrupted value that should be skipped."""
	if not name:
		return True
	low = name.lower().strip()
	# Filter out obvious placeholders and corruption artifacts
	skip_patterns = [
		r"^/?provider$",  # /Provider or Provider alone
		r"^/?receiver$",  # /Receiver or Receiver alone
		r"^/?supplier$",  # /Supplier or Supplier alone
		r"^/?agent$",  # /Agent or Agent alone
		r"^/?corporate",  # /Corporate or Corporate alone
		r"^/?service",  # /Service or Service alone
		r"^/$",  # Just a slash
		r"^[/\-\s]+$",  # Only slashes, dashes, spaces
		r"^\d+$",  # Only numbers
	]
	for pattern in skip_patterns:
		if re.match(pattern, low):
			return True
	# If name starts with "/" or is very short, likely corrupted
	if name.startswith("/") or len(low) < 3:
		return True
	return False


def extract_nkgsb_amount_row(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable, cgst, sgst, igst, total from NKGSB style 'Amount' row."""
	processed = re.sub(r"\s+", " ", text)
	if "nkgsb" not in processed.lower():
		return "", "", "", "", ""

	# Flexible OCR-tolerant header: CGST can appear as ICGST, spacing around @ may vary.
	pattern = (
		r"taxable\s*value\s+i?cgst\s*@?\s*\d{1,2}\s+sgst/utgst\s*@?\s*\d{1,2}\s+igst\s*@?\s*\d{1,2}\s+total"
		r".*?amount\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)"
	)
	match = re.search(pattern, processed, re.IGNORECASE)
	if not match:
		# Last fallback: take first 5 amount-like numbers after "Amount".
		fallback = re.search(r"amount\s+((?:[0-9][0-9,]*(?:\.[0-9]{1,2})?\s+){4}[0-9][0-9,]*(?:\.[0-9]{1,2})?)", processed, re.IGNORECASE)
		if not fallback:
			return "", "", "", "", ""
		parts = [clean_amount(v) for v in re.findall(r"[0-9][0-9,]*(?:\.[0-9]{1,2})?", fallback.group(1))]
		if len(parts) < 5:
			return "", "", "", "", ""
		return parts[0], parts[1], parts[2], parts[3], parts[4]

	values = [clean_amount(match.group(i)) for i in range(1, 6)]
	if any(not v for v in values):
		return "", "", "", "", ""
	return values[0], values[1], values[2], values[3], values[4]


def extract_jk_bank_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable, CGST, SGST/UTGST, IGST, total for Jammu & Kashmir Bank OCR text."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if ("jammu" not in low or "kashmir" not in low) and "jammo" not in low:
		return "", "", "", "", ""

	def normalize_tax_like_amount(raw: str) -> str:
		val = clean_amount(raw)
		if not val:
			return ""
		try:
			n = float(val)
		except (ValueError, TypeError):
			return ""
		# OCR often drops decimal separators (e.g. 1547163 -> 15471.63).
		if "." not in val and n >= 100000:
			n = n / 100.0
		return f"{n:.2f}"

	taxable = clean_amount(find_first([
		r"taxable\s*val(?:ue|ua|\w*)\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"value\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*taxable\s*val",
	], processed))

	cgst = ""
	sgst = ""
	igst = ""

	# Capture percentage-amount pairs near tax-payable block, tolerant to OCR spellings.
	segment_match = re.search(r"tax\s*pay\w*.*?(?:total\s*amount\s*p\w*)", processed, re.IGNORECASE)
	segment = segment_match.group(0) if segment_match else processed
	pairs = re.findall(r"(\d{1,2}(?:\.\d+)?)\s*%\|?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", segment)
	state_vals: List[str] = []
	igst_vals: List[str] = []
	for rate_raw, amt_raw in pairs:
		amt = clean_amount(amt_raw)
		if not amt:
			continue
		try:
			rate = float(rate_raw)
			amt_num = float(amt)
		except (ValueError, TypeError):
			continue
		if amt_num <= 10:
			continue
		if 8.0 <= rate <= 10.5:
			state_vals.append(amt)
		elif 17.0 <= rate <= 19.5:
			igst_vals.append(amt)

	# OCR fallback where tax labels are degraded (CEST/SEST etc.) or decimals are dropped.
	if not state_vals and not igst_vals:
		row_match = re.search(
			r"taxable\s*value.*?9(?:\.0+)?\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(?:s|sg|s/?ut|se?st)[^0-9]{0,20}9(?:\.0+)?\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
			processed,
			re.IGNORECASE,
		)
		if row_match:
			c1 = normalize_tax_like_amount(row_match.group(1))
			c2 = normalize_tax_like_amount(row_match.group(2))
			if c1:
				state_vals.append(c1)
			if c2:
				state_vals.append(c2)

	if len(state_vals) >= 2:
		cgst, sgst = state_vals[0], state_vals[1]
	elif len(state_vals) == 1:
		cgst = state_vals[0]
		sgst = state_vals[0]
	elif igst_vals:
		igst = igst_vals[0]

	total = ""
	total_anchor = re.search(
		r"(?:total\s*amount\s*p\w*|total\s*invoice\s*value|invoice\s*value|total\s*amount)",
		processed,
		re.IGNORECASE,
	)
	if total_anchor:
		tail = processed[total_anchor.end(): total_anchor.end() + 160]
		nums: List[float] = []
		for match in re.finditer(r"[0-9][0-9,]*(?:\.[0-9]{1,2})?", tail):
			end = match.end()
			if end < len(tail) and tail[end:end + 1] == "%":
				continue
			parsed = clean_amount(match.group(0))
			if not parsed:
				continue
			try:
				val = float(parsed)
			except (ValueError, TypeError):
				continue
			if val > 10:
				nums.append(val)
		if nums:
			total = f"{max(nums):.2f}"

	# Prefer explicit total-in-words when OCRed numeric total is noisy.
	total_words = find_first([
		r"total\s*in\s*words\s*rupees\s*([a-z\s-]{8,220})",
	], processed)
	if total_words:
		words_num = words_to_number(total_words)
		if words_num:
			parsed = clean_amount(words_num)
			if parsed:
				total = parsed

	# Fallback: derive total from components when OCR label anchor is weak.
	if not total and taxable:
		try:
			taxable_num = float(taxable)
			cgst_num = float(cgst) if cgst else 0.0
			sgst_num = float(sgst) if sgst else 0.0
			igst_num = float(igst) if igst else 0.0
			derived = taxable_num + cgst_num + sgst_num + igst_num
			if derived > taxable_num:
				total = f"{derived:.2f}"
		except (ValueError, TypeError):
			pass

	# If taxable is noisy but total and state taxes are strong, derive taxable deterministically.
	try:
		taxable_num = float(taxable.replace(",", "")) if taxable else 0.0
		total_num = float(total.replace(",", "")) if total else 0.0
		cgst_num = float(cgst.replace(",", "")) if cgst else 0.0
		sgst_num = float(sgst.replace(",", "")) if sgst else 0.0
		if total_num > 0 and cgst_num > 0 and sgst_num > 0:
			derived_taxable = total_num - (cgst_num + sgst_num)
			if derived_taxable > 0 and (taxable_num <= 0 or abs(taxable_num - derived_taxable) > max(10.0, derived_taxable * 0.25)):
				taxable = f"{derived_taxable:.2f}"
	except (ValueError, TypeError):
		pass

	return taxable, cgst, sgst, igst, total


def extract_india_post_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable, CGST, SGST, IGST, total for India Post Payments Bank layout."""
	processed = re.sub(r"\s+", " ", text)
	if "india post payments bank" not in processed.lower():
		return "", "", "", "", ""

	total = clean_amount(find_first([
		r"invoice\s*total\s*amount\s*[-:]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	segment_match = re.search(r"description\s+of\s+services.*?invoice\s*total\s*amount", processed, re.IGNORECASE)
	segment = segment_match.group(0) if segment_match else processed

	taxable = clean_amount(find_first([
		r"taxable\s*value[^0-9]{0,20}([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+9\s*%",
		r"\b([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+9\s*%\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?\s+9\s*%",
	], segment))

	pairs = re.findall(r"(9|18)\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", segment, re.IGNORECASE)
	nines = [clean_amount(a) for rate, a in pairs if rate == "9" and clean_amount(a)]
	eighteens = [clean_amount(a) for rate, a in pairs if rate == "18" and clean_amount(a)]

	cgst = nines[-2] if len(nines) >= 2 else (nines[0] if len(nines) == 1 else "")
	sgst = nines[-1] if len(nines) >= 2 else (nines[0] if len(nines) == 1 else "")
	igst = eighteens[-1] if eighteens else ""

	# Prefer either IGST-only or CGST+SGST based on non-zero values.
	try:
		cgst_num = float(cgst.replace(",", "")) if cgst else 0
		sgst_num = float(sgst.replace(",", "")) if sgst else 0
		igst_num = float(igst.replace(",", "")) if igst else 0
		if igst_num > 0 and (cgst_num + sgst_num) == 0:
			cgst = ""
			sgst = ""
		elif (cgst_num + sgst_num) > 0 and igst_num == 0:
			igst = ""
		elif igst_num > 0 and (cgst_num + sgst_num) > 0 and total and taxable:
			total_num = float(total.replace(",", ""))
			taxable_num = float(taxable.replace(",", "")) if taxable else 0
			gst_from_total = max(0.0, total_num - taxable_num)
			if abs(gst_from_total - igst_num) <= abs(gst_from_total - (cgst_num + sgst_num)):
				cgst = ""
				sgst = ""
			else:
				igst = ""
	except (ValueError, TypeError):
		pass

	return taxable, cgst, sgst, igst, total


def extract_dhanlaxmi_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from Dhanlaxmi Bank OCR tax invoices."""
	processed = re.sub(r"\s+", " ", text)
	if "dhanlaxmi" not in processed.lower():
		return "", "", "", "", ""

	taxable = clean_amount(find_first([
		r"taxable\s*value\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))
	cgst = clean_amount(find_first([
		r"cgst\s*[:|]?\s*9(?:\.0+)?\s*%\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))
	sgst = clean_amount(find_first([
		r"sgst\s*[:|]?\s*9(?:\.0+)?\s*%\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))
	igst = clean_amount(find_first([
		r"igst\s*[:|]?\s*18(?:\.0+)?\s*%\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	total = ""
	tail_match = re.search(r"total\s*amount\s*payable\s*[:|]?\s*(.{0,80})", processed, re.IGNORECASE)
	if tail_match:
		nums = [clean_amount(n) for n in re.findall(r"[0-9][0-9,]*(?:\.[0-9]{1,2})?", tail_match.group(1))]
		nums = [n for n in nums if n]
		vals: List[float] = []
		for n in nums:
			try:
				vals.append(float(n))
			except (TypeError, ValueError):
				continue
		if vals:
			total = f"{max(vals):.2f}"

	if not total:
		total = clean_amount(find_first([
			r"total\s*amount\s*payable\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		], processed))

	return taxable, cgst, sgst, igst, total


def extract_city_union_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from City Union Bank table rows."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "city union bank" not in low:
		return "", "", "", "", ""

	# Prefer summary labels when available.
	taxable = clean_amount(find_first([
		r"total\s*amount\s*before\s*tax\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"taxable\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))
	cgst = clean_amount(find_first([
		r"add\s*[:\-]?\s*cgst\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))
	sgst = clean_amount(find_first([
		r"add\s*[:\-]?\s*sgst\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))
	igst = clean_amount(find_first([
		r"add\s*[:\-]?\s*igst\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"igst\s*[:|]?\s*(?:18(?:\.00)?\s*%\s*)?([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))
	total = clean_amount(find_first([
		r"total\s*amount\s*after\s*tax\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"grand\s*total\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	# Prefer the first invoice table block over annexure rows.
	main_block_match = re.search(r"sl\s*no.*?amount\s*in\s*words", processed, re.IGNORECASE)
	main_block = main_block_match.group(0) if main_block_match else ""
	if main_block:
		igst_row = re.search(
			r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+0\.00\s+0\.00\s+0\.00\s+0\.00\s+18(?:\.00)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
			main_block,
			re.IGNORECASE,
		)
		if igst_row:
			taxable = taxable or clean_amount(igst_row.group(1))
			igst = clean_amount(igst_row.group(2))
			total = total or clean_amount(igst_row.group(3))
			cgst = ""
			sgst = ""

	# Fallback to 'Total ... taxable/cgst/sgst/igst/total' row.
	if not (taxable and total):
		for m in re.finditer(r"total\s+([0-9.,\s]{20,120})", processed, re.IGNORECASE):
			nums = [clean_amount(n) for n in re.findall(r"[0-9][0-9,]*(?:\.[0-9]{1,2})?", m.group(1))]
			nums = [n for n in nums if n]
			if len(nums) < 6:
				continue
			try:
				disc, txb, cg, sg, ig, ttl = [float(n) for n in nums[:6]]
			except (TypeError, ValueError):
				continue
			if txb > 0 and ttl > txb and (cg > 0 or sg > 0 or ig > 0):
				taxable = taxable or f"{txb:.2f}"
				if not cgst and cg > 0:
					cgst = f"{cg:.2f}"
				if not sgst and sg > 0:
					sgst = f"{sg:.2f}"
				if not igst and ig > 0:
					igst = f"{ig:.2f}"
				total = total or f"{ttl:.2f}"
				break

	# Treat explicit zero IGST as empty value.
	try:
		if igst and float(igst) <= 0:
			igst = ""
	except (TypeError, ValueError):
		pass

	# If we have taxable + IGST but missing/weak total, derive deterministic total.
	try:
		taxable_num = float(taxable.replace(",", "")) if taxable else 0.0
		igst_num = float(igst.replace(",", "")) if igst else 0.0
		total_num = float(total.replace(",", "")) if total else 0.0
		if taxable_num > 0 and igst_num > 0 and (total_num <= taxable_num):
			total = f"{(taxable_num + igst_num):.2f}"
	except (TypeError, ValueError):
		pass

	return taxable, cgst, sgst, igst, total


def extract_probitas_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from Probitas invoice layouts."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "probitas insurance brokers" not in low:
		return "", "", "", "", ""

	taxable = clean_amount(find_first([
		r"hsn/sac\s+taxable\s*value\s+[0-9,.\s%]{6,80}?([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"taxable\s*value\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))
	cgst = ""
	sgst = ""
	igst = ""
	total = clean_amount(find_first([
		r"amount\s*chargeable\s*\(in\s*words\).*?total\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"\btotal\b\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	# Common state-tax row: <tax_total> <cgst> 9% <sgst> 9% <taxable>
	m_state = re.search(
		r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*9\s*%\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*9\s*%\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		processed,
		re.IGNORECASE,
	)
	if m_state:
		cgst = clean_amount(m_state.group(2))
		sgst = clean_amount(m_state.group(3))
		taxable = taxable or clean_amount(m_state.group(4))

	# IGST variant: <tax_total> 18% <igst> <taxable>
	m_igst = re.search(
		r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*18\s*%\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		processed,
		re.IGNORECASE,
	)
	if m_igst and not (cgst and sgst):
		igst = clean_amount(m_igst.group(2))
		taxable = taxable or clean_amount(m_igst.group(3))

	try:
		taxable_num = float((taxable or "").replace(",", "")) if taxable else 0.0
		cgst_num = float((cgst or "").replace(",", "")) if cgst else 0.0
		sgst_num = float((sgst or "").replace(",", "")) if sgst else 0.0
		igst_num = float((igst or "").replace(",", "")) if igst else 0.0
		if taxable_num <= 0 and cgst_num > 0 and sgst_num > 0:
			if abs(cgst_num - sgst_num) <= max(2.0, max(cgst_num, sgst_num) * 0.05):
				taxable_num = (cgst_num + sgst_num) / 0.18
				taxable = f"{taxable_num:.2f}"
		gst_num = igst_num if igst_num > 0 else (cgst_num + sgst_num)
		total_num = float((total or "").replace(",", "")) if total else 0.0
		if taxable_num > 0 and gst_num > 0 and (total_num <= taxable_num or abs(total_num - (taxable_num + gst_num)) > max(2.0, (taxable_num + gst_num) * 0.05)):
			total = f"{(taxable_num + gst_num):.2f}"
	except (ValueError, TypeError):
		pass

	return taxable, cgst, sgst, igst, total


def extract_ethika_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from Ethika invoice layouts."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "ethika insurance" not in low:
		return "", "", "", "", ""

	taxable = clean_amount(find_first([
		r"sub\s*total\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"hsn/sac\s+rate\s+igst\s+amount\s+\d+\s+brokerage\s+income\s+\d{4,8}\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	igst = clean_amount(find_first([
		r"igst\s*\d{1,2}\s*\(?\d{1,2}\s*%\)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"\b18\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\b",
	], processed))

	total = clean_amount(find_first([
		r"balance\s*due\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"\btotal\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	try:
		taxable_num = float(taxable.replace(",", "")) if taxable else 0.0
		igst_num = float(igst.replace(",", "")) if igst else 0.0
		total_num = float(total.replace(",", "")) if total else 0.0
		if taxable_num > 0 and igst_num > 0:
			expected = taxable_num + igst_num
			if total_num <= taxable_num or abs(total_num - taxable_num) <= max(2.0, taxable_num * 0.03):
				total = f"{expected:.2f}"
	except (TypeError, ValueError):
		pass

	return taxable, "", "", igst, total


def extract_catalyst_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from Catalyst invoice OCR text."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "catalyst" not in low:
		return "", "", "", "", ""

	taxable = clean_amount(find_first([
		r"total\s*taxable\s*value\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"\b([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*18\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	igst = clean_amount(find_first([
		r"igst\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"\b[0-9][0-9,]*(?:\.[0-9]{1,2})?\s*18\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	total = clean_amount(find_first([
		r"amount\s*chargeable[^0-9]{0,40}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"grand\s*total\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	try:
		taxable_num = float(taxable.replace(",", "")) if taxable else 0.0
		igst_num = float(igst.replace(",", "")) if igst else 0.0
		total_num = float(total.replace(",", "")) if total else 0.0
		if taxable_num > 0 and igst_num > 0 and total_num <= taxable_num:
			total = f"{(taxable_num + igst_num):.2f}"
	except (TypeError, ValueError):
		pass

	return taxable, "", "", igst, total


def extract_incred_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from Incred invoice layouts."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "incred financial" not in low:
		return "", "", "", "", ""

	# Typical row: taxable cgst sgst Total <total>
	for m in re.finditer(
		r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+total\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		processed,
		re.IGNORECASE,
	):
		taxable = clean_amount(m.group(1))
		cgst = clean_amount(m.group(2))
		sgst = clean_amount(m.group(3))
		total = clean_amount(m.group(4))
		try:
			tx = float(taxable)
			cg = float(cgst)
			sg = float(sgst)
			tt = float(total)
			if tx > 100 and cg > 10 and sg > 10 and abs(tt - (tx + cg + sg)) <= max(2.0, tt * 0.03):
				return taxable, cgst, sgst, "", total
		except (TypeError, ValueError):
			continue

	return "", "", "", "", ""


def extract_jb_boda_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from JB Boda invoice layouts."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "j.b.boda" not in low and "jb boda" not in low:
		return "", "", "", "", ""

	taxable = clean_amount(find_first([
		r"brokerage[^0-9]{0,40}([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*18\s*%\s*997161",
		r"taxable[^0-9]{0,40}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	cgst = clean_amount(find_first([
		r"central\s*gst\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))
	sgst = clean_amount(find_first([
		r"state\s*gst\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	total = clean_amount(find_first([
		r"total\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*amount\s*chargeable",
		r"amount\s*chargeable[^0-9]{0,40}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	try:
		tx = float(taxable) if taxable else 0.0
		cg = float(cgst) if cgst else 0.0
		sg = float(sgst) if sgst else 0.0
		tt = float(total) if total else 0.0
		if tx > 0 and cg > 0 and sg > 0 and tt <= 0:
			total = f"{(tx + cg + sg):.2f}"
	except (TypeError, ValueError):
		pass

	return taxable, cgst, sgst, "", total


def extract_ideal_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from Ideal Insurance invoice tables."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "ideal insurance" not in low:
		return "", "", "", "", ""

	# Intra-state rows: 997161 <taxable> 9 <cgst> 9 <sgst> <total>
	for m in re.finditer(
		r"9971\d{2}\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+9(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+9(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		processed,
		re.IGNORECASE,
	):
		taxable = clean_amount(m.group(1))
		cgst = clean_amount(m.group(2))
		sgst = clean_amount(m.group(3))
		total = clean_amount(m.group(4))
		try:
			tx = float(taxable)
			cg = float(cgst)
			sg = float(sgst)
			tt = float(total)
			if tx > 10 and cg > 0 and sg > 0 and abs(tt - (tx + cg + sg)) <= max(3.0, tt * 0.03):
				return taxable, cgst, sgst, "", total
		except (ValueError, TypeError):
			continue

	# Inter-state rows: 997161 <taxable> 18 <igst> <total>
	for m in re.finditer(
		r"9971\d{2}\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+18(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		processed,
		re.IGNORECASE,
	):
		taxable = clean_amount(m.group(1))
		igst = clean_amount(m.group(2))
		total = clean_amount(m.group(3))
		try:
			tx = float(taxable)
			ig = float(igst)
			tt = float(total)
			if tx > 10 and ig > 0 and abs(tt - (tx + ig)) <= max(3.0, tt * 0.03):
				return taxable, "", "", igst, total
		except (ValueError, TypeError):
			continue

	return "", "", "", "", ""


def extract_mahindra_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from Mahindra Insurance Brokers layouts."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "mahindra insurance brokers" not in low:
		return "", "", "", "", ""

	# Typical row:
	# 997161 1 <taxable> <...> <...> C : 9 S/UT : 9 C : <cgst> S/UT : <sgst>
	for m in re.finditer(
		r"9971\d{2}\s+1\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)(?:\s+[0-9][0-9,]*(?:\.[0-9]{1,2})?){1,3}\s+C\s*:\s*9\s*S/UT\s*:\s*9\s*C\s*:\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*S/UT\s*:\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		processed,
		re.IGNORECASE,
	):
		taxable = clean_amount(m.group(1))
		cgst = clean_amount(m.group(2))
		sgst = clean_amount(m.group(3))
		try:
			tx = float(taxable)
			cg = float(cgst)
			sg = float(sgst)
			if tx > 10 and cg > 0 and sg > 0 and abs(cg - sg) <= max(cg, sg) * 0.2:
				total = f"{(tx + cg + sg):.2f}"
				return taxable, cgst, sgst, "", total
		except (ValueError, TypeError):
			continue

	return "", "", "", "", ""


def extract_bajaj_housing_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from Bajaj Housing Finance invoice tables."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "bajaj housing finance" not in low:
		return "", "", "", "", ""

	# Typical row near HSN 997161: taxable 9 cgst 9 sgst total
	for m in re.finditer(
		r"([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?)\s+9(?:\.0+)?\s+([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?)\s+9(?:\.0+)?\s+([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?)\s+([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?)",
		processed,
		re.IGNORECASE,
	):
		taxable = clean_amount(m.group(1))
		cgst = clean_amount(m.group(2))
		sgst = clean_amount(m.group(3))
		total = clean_amount(m.group(4))
		try:
			tx = float(taxable)
			cg = float(cgst)
			sg = float(sgst)
			tt = float(total)
			if tx > 100 and cg > 10 and sg > 10 and abs(tt - (tx + cg + sg)) <= max(3.0, tt * 0.02):
				return taxable, cgst, sgst, "", total
		except (ValueError, TypeError):
			continue

	return "", "", "", "", ""


def extract_coverkraft_amounts(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from Coverkraft invoice layout."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "coverkraft" not in low:
		return "", "", "", "", ""

	taxable = clean_amount(find_first([
		r"net\s*amount\s*\(rs\.?\)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"sub\s*total\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	cgst = clean_amount(find_first([
		r"cgst\s*@\s*9%\s*9%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"cgst\s*@\s*9%[^0-9]{0,20}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	sgst = clean_amount(find_first([
		r"sgst\s*@\s*9%\s*9%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"sgst\s*@\s*9%[^0-9]{0,20}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	total = clean_amount(find_first([
		r"gross\s*total\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	try:
		tx = float(taxable) if taxable else 0.0
		cg = float(cgst) if cgst else 0.0
		sg = float(sgst) if sgst else 0.0
		tt = float(total) if total else 0.0
		if tx > 0 and cg > 0 and sg > 0:
			expected = tx + cg + sg
			if tt <= 0 or abs(tt - expected) > max(2.0, expected * 0.03):
				total = f"{expected:.2f}"
			return taxable, cgst, sgst, "", total
	except (ValueError, TypeError):
		pass

	return "", "", "", "", ""


def apply_party_overrides(row: Dict[str, str], source_path: Path, context_text: str) -> Dict[str, str]:
	source_low = str(source_path).lower()
	agent_name = row.get("Agent Name", "") or ""
	service_recipient = row.get("Name of Service Receipient", "") or ""
	provider_name, receiver_name = extract_provider_receiver_names(context_text)
	if provider_name:
		agent_name = provider_name
	if receiver_name:
		service_recipient = normalize_balic_service_recipient_name(receiver_name, context_text)

	inferred_agent = infer_corporate_agent_from_context(source_path, context_text)
	if inferred_agent and (not agent_name or _is_placeholder_agent_name(agent_name) or agent_name.strip().lower() == "imf"):
		agent_name = inferred_agent
	else:
		agent_name = normalize_corporate_agent_name(sanitize_party_name(agent_name))
	if agent_name and agent_name.strip().lower() == "bajaj auto":
		agent_name = "Bajaj Housing Finance Limited"

	# Hard role rule: BALIC entities must be service recipients, never agents.
	if is_known_balic_entity(agent_name):
		if not service_recipient:
			service_recipient = agent_name
		agent_name = ""

	row["Agent Name"] = agent_name
	row["Name of Service Receipient"] = normalize_balic_service_recipient_name(service_recipient, context_text)

	return row


def normalize_mapping_anomalies(row: Dict[str, str], context_text: str = "", source_path: Optional[Path] = None) -> Dict[str, str]:
	"""Fix known post-extraction mapping anomalies at row level."""
	# Enforce strict GSTIN validity for final output fields.
	for gstin_col in ["BALIC GSTN", "BROKER GSTN"]:
		gstin_val = (row.get(gstin_col, "") or "").strip().upper()
		if gstin_val and not is_valid_gstin(gstin_val):
			row[gstin_col] = ""
		elif gstin_val:
			row[gstin_col] = gstin_val

	if not (row.get("BALIC STATE", "") or "").strip() and (row.get("BALIC GSTN", "") or "").strip():
		row["BALIC STATE"] = gstin_to_state_name(row.get("BALIC GSTN", ""))

	if not (row.get("BROKER GSTN STATE", "") or "").strip() and (row.get("BROKER GSTN", "") or "").strip():
		row["BROKER GSTN STATE"] = gstin_to_state_name(row.get("BROKER GSTN", ""))

	row = apply_company_wise_issue_mapping(row, source_path, context_text)

	# Generic fallback: use source name/text to recover a missing or placeholder agent.
	if ((not (row.get("Agent Name", "") or "").strip()) or _is_placeholder_agent_name(row.get("Agent Name", ""))) and source_path:
		source_agent = infer_corporate_agent_from_context(source_path, context_text)
		if source_agent:
			row["Agent Name"] = source_agent

	if (row.get("Agent Name", "") or "").strip().lower() == "bajaj auto":
		row["Agent Name"] = "Bajaj Housing Finance Limited"

	if AGENT_CODE_BY_NAME and not (row.get("AGENT_CODE", "") or "").strip():
		candidate_names = [
			row.get("Agent Name", ""),
			str(source_path or ""),
			Path(source_path).name if source_path else "",
		]
		for candidate_name in candidate_names:
			candidate_name = (candidate_name or "").strip()
			if not candidate_name:
				continue
			matched_code, clean_name, similarity = find_best_matching_agent_code(candidate_name)
			if matched_code:
				row["AGENT_CODE"] = matched_code
				if clean_name:
					row["Agent Name"] = clean_name
				break

	balic_gstn = (row.get("BALIC GSTN", "") or "").strip().upper()
	broker_gstn = (row.get("BROKER GSTN", "") or "").strip().upper()
	agent_pan = (row.get("Agent PAN", "") or "").strip().upper()
	recipient_low = (row.get("Name of Service Receipient", "") or "").strip().lower()
	source_low = str(source_path).lower() if source_path else ""
	ctx_low = (context_text or "").lower()

	if (
		balic_gstn
		and re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", agent_pan)
		and agent_pan in balic_gstn
		and ("bajaj" in recipient_low or "allianz" in recipient_low or "life" in recipient_low)
	):
		candidates = extract_gstin_candidates(context_text or "")
		for candidate in candidates:
			cand = (candidate or "").strip().upper()
			if is_valid_gstin(cand) and cand != balic_gstn and agent_pan not in cand:
				row["BALIC GSTN"] = cand if "AADCA1701E" in cand else row["BALIC GSTN"]
				break

	if balic_gstn:
		row["BALIC STATE"] = gstin_to_state_name(balic_gstn)
	if broker_gstn:
		row["BROKER GSTN STATE"] = gstin_to_state_name(broker_gstn)

	if balic_gstn and broker_gstn and balic_gstn == broker_gstn:
		candidates = extract_gstin_candidates(context_text or "")
		replacement = ""
		for candidate in candidates:
			cand = (candidate or "").strip().upper()
			if not is_valid_gstin(cand) or cand == balic_gstn:
				continue
			if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", agent_pan) and agent_pan in cand:
				replacement = cand
				break
		if not replacement:
			for candidate in candidates:
				cand = (candidate or "").strip().upper()
				if is_valid_gstin(cand) and cand != balic_gstn and "AADCA1701E" not in cand:
					replacement = cand
					break
		if replacement:
			row["BROKER GSTN"] = replacement
			row["BROKER GSTN STATE"] = gstin_to_state_name(replacement)

	row["Name of Service Receipient"] = normalize_balic_service_recipient_name(row.get("Name of Service Receipient", ""), context_text)

	if not (row.get("BALIC STATE", "") or "").strip() and source_path:
		for state in sorted(INDIAN_STATES, key=len, reverse=True):
			if re.search(rf"\b{re.escape(state)}\b", str(source_path), re.IGNORECASE):
				row["BALIC STATE"] = state
				break

	vendor_date = (row.get("Vendor Inv Date", "") or "").strip()
	if not vendor_date or not re.fullmatch(r"\d{2}/\d{2}/20(?:25|26)", vendor_date):
		inferred_date = infer_plausible_date_from_text(f"{source_path} {context_text}")
		if inferred_date:
			row["Vendor Inv Date"] = inferred_date

	# Dhanlaxmi fallback: if broker GSTIN is missing, recover any valid non-BALIC GSTIN from context.
	if "dhanlaxmi" in source_low or "dhanlaxmi" in ctx_low:
		if not (row.get("BROKER GSTN", "") or "").strip():
			for candidate in extract_gstin_candidates(context_text or ""):
				cand = candidate.strip().upper()
				if is_valid_gstin(cand) and "AADCA1701E" not in cand:
					row["BROKER GSTN"] = cand
					row["BROKER GSTN STATE"] = gstin_to_state_name(cand)
					break

	# For City Union and Bajaj receipts: infer state from path/text (Tamil Nadu vs Telangana)
	if any(k in source_low for k in ("city union", "bajaj")) or any(k in ctx_low for k in ("city union", "bajaj")):
		inferred_state = ""
		# Prefer explicit Tamil Nadu indicators first
		if any(tok in source_low for tok in ("tamil nadu", "tamilnadu", "tamil", " tn ")) or any(tok in ctx_low for tok in ("tamil nadu", "tamilnadu", "tamil", " tn ")):
			inferred_state = "Tamil Nadu"
		# Check Telangana indicators (including common misspelling)
		elif any(tok in source_low for tok in ("telangana", "telengana", "hyderabad", "hyd")) or any(tok in ctx_low for tok in ("telangana", "telengana", "hyderabad", "hyd")):
			inferred_state = "Telangana"

		# If we inferred a state, prefer GSTIN candidates whose state matches it.
		if inferred_state:
			row_state_before = (row.get("BALIC STATE", "") or "").strip()
			if not row_state_before:
				row["BALIC STATE"] = inferred_state

			if not (row.get("BALIC GSTN", "") or "").strip():
				for candidate in extract_gstin_candidates(context_text or ""):
					cand = candidate.strip().upper()
					if not is_valid_gstin(cand):
						continue
					cand_state = gstin_to_state_name(cand)
					if cand_state and cand_state == inferred_state:
						# prefer BALIC matching PAN fragment if present
						if "AADCA1701E" in cand or not (row.get("BALIC GSTN", "") or "").strip():
							row["BALIC GSTN"] = cand
							row["BALIC STATE"] = cand_state
							break

			if not (row.get("BROKER GSTN", "") or "").strip():
				for candidate in extract_gstin_candidates(context_text or ""):
					cand = candidate.strip().upper()
					if not is_valid_gstin(cand):
						continue
					cand_state = gstin_to_state_name(cand)
					if cand_state and cand_state == inferred_state and "AADCA1701E" not in cand:
						row["BROKER GSTN"] = cand
						row["BROKER GSTN STATE"] = cand_state
						break

	# City Union and other state-tax rows: if GST TOTAL matches IGST and state tax is also present, prefer IGST-only.
	try:
		gst_total = _amount_to_float(row.get("GST TOTAL AMT", ""))
		igst_v = _amount_to_float(row.get("IGST", ""))
		cgst_v = _amount_to_float(row.get("CGST @ 9%", ""))
		sgst_v = _amount_to_float(row.get("SGST @ 9%", ""))
		utgst_v = _amount_to_float(row.get("UTGST", ""))
		# If GST TOTAL equals IGST within tolerance, but state components are present, prefer IGST-only.
		if gst_total > 0 and igst_v > 0 and abs(gst_total - igst_v) <= max(2.0, igst_v * 0.02) and (cgst_v > 0 or sgst_v > 0 or utgst_v > 0):
			row["CGST @ 9%"] = ""
			row["SGST @ 9%"] = ""
			row["UTGST"] = ""
			row["IGST"] = f"{igst_v:.2f}"
			row["GST TOTAL AMT"] = f"{igst_v:.2f}"

		# If GST TOTAL is approximately double the IGST, the invoice likely represents split state tax (CGST+SGST).
		# In that case map IGST -> CGST/SGST halves and clear IGST.
		if gst_total > 0 and igst_v > 0 and abs(gst_total - (2.0 * igst_v)) <= max(2.0, igst_v * 0.02) and cgst_v == 0 and sgst_v == 0 and utgst_v == 0:
			half = igst_v
			row["CGST @ 9%"] = f"{half:.2f}"
			row["SGST @ 9%"] = f"{half:.2f}"
			row["UTGST"] = ""
			row["IGST"] = ""
			row["GST TOTAL AMT"] = f"{gst_total:.2f}"
	except (ValueError, TypeError):
		pass

	# Clean invoice number text.
	inv = (row.get("Vendor Inv No", "") or "").strip()
	if inv:
		row["Vendor Inv No"] = clean_invoice_no(inv)

	if "GROUPNONMICRO" in (row.get("Narration", "") or ""):
		row["Narration"] = (row.get("Narration", "") or "").replace("GROUPNONMICRO", "").strip()

	# Ujjivan: set agent name and code if missing
	if "ujjivan" in source_low or "ujjivan" in ctx_low:
		if not (row.get("Agent Name", "") or "").strip():
			row["Agent Name"] = "Ujjivan"
		# Attempt direct extraction of agent code from context (e.g., 'Agent Code: ABC123').
		if not (row.get("AGENT_CODE", "") or "").strip():
			m = re.search(r"agent\s*code\s*[:\-]?\s*([A-Z0-9-]{3,20})", context_text or "", re.IGNORECASE)
			if not m:
				m = re.search(r"agent\s*code\s*[:\-]?\s*([A-Z0-9-]{3,20})", str(source_path) or "", re.IGNORECASE)
			if m:
				row["AGENT_CODE"] = m.group(1).upper()
			else:
				if AGENT_CODE_BY_NAME:
					matched, clean_name, sim = find_best_matching_agent_code(row.get("Agent Name", ""))
					if matched:
						row["AGENT_CODE"] = matched
						row["Agent Name"] = clean_name
				# fallback: try fuzzy match using source path segments
				if not (row.get("AGENT_CODE", "") or "").strip() and AGENT_CODE_BY_NAME:
					parts = re.split(r"[\\/\-_ ]+", str(source_path) or "")
					for p in parts:
						if not p:
							continue
						matched, clean_name, sim = find_best_matching_agent_code(p)
						if matched:
							row["AGENT_CODE"] = matched
							row["Agent Name"] = clean_name
							break
				if not (row.get("AGENT_CODE", "") or "").strip() and AGENT_CODE_BY_NAME:
					for agent_name_key, agent_code in AGENT_CODE_BY_NAME.items():
						if "ujjivan" in agent_name_key:
							row["AGENT_CODE"] = agent_code
							row["Agent Name"] = AGENT_NAME_BY_CODE.get(agent_code, "Ujjivan")
							break

	# IMF: set agent name if missing (IMF files don't have embedded company names usually)
	if "imf" in source_low or "imf" in ctx_low:
		if not (row.get("Agent Name", "") or "").strip():
			row["Agent Name"] = "IMF"

	# Bajaj Housing detection: look for common variants in text or filename and set Agent Name.
	if any(k in source_low for k in ("bajaj housing", "bajaj housing finance", "bajaj housing finance limited")) or any(k in ctx_low for k in ("bajaj housing", "bajaj housing finance", "bajaj housing finance limited")):
		if not (row.get("Agent Name", "") or "").strip():
			row["Agent Name"] = "Bajaj Housing Finance Limited"
		# attempt to set agent code via mapping or context
		if not (row.get("AGENT_CODE", "") or "").strip():
			if AGENT_CODE_BY_NAME:
				matched, clean_name, sim = find_best_matching_agent_code(row.get("Agent Name", ""))
				if matched:
					row["AGENT_CODE"] = matched

	return row



def extract_state_from_text(flattened: str) -> str:
	patterns = [
		r"State\s+State\s+Code\s+([A-Za-z ]{3,30})\s+\d{2}",
		r"(?:state\s*[:\-]?\s*)([A-Za-z][A-Za-z\s]{2,40})",
		r"(?:place\s*of\s*supply\s*[:\-]?\s*)([A-Za-z][A-Za-z\s]{2,40})",
		r"(?:location\s*[:\-]?\s*)([A-Za-z][A-Za-z\s]{2,40})",
	]
	for pattern in patterns:
		match = re.search(pattern, flattened, re.IGNORECASE)
		if match:
			state = normalize_state_name(match.group(1))
			if state:
				return state

	for state in INDIAN_STATES:
		if re.search(rf"\b{re.escape(state)}\b", flattened, re.IGNORECASE):
			return state

	return ""


def extract_company_from_address_start(flattened: str) -> str:
	patterns = [
		r"(?:name\s*&?\s*address\s*[:\-]?\s*)([A-Za-z][A-Za-z0-9\s&.,()'/-]{8,120}?)(?:\s+(?:near|plot|road|street|city|state|pin|gstin|phone|contact|cin)\b)",
		r"(?:address\s*[:\-]?\s*)([A-Za-z][A-Za-z0-9\s&.,()'/-]{8,120}?)(?:\s+(?:near|plot|road|street|city|state|pin|gstin|phone|contact|cin)\b)",
	]
	for pattern in patterns:
		match = re.search(pattern, flattened, re.IGNORECASE)
		if match:
			name = re.sub(r"\s+", " ", match.group(1)).strip(" ,.-")
			if len(name) >= 8:
				return normalize_balic_company_name(name)
	return ""


def is_valid_invoice_no(value: str) -> bool:
	candidate = clean_invoice_no(value)
	if len(candidate) < 4:
		return False
	if not re.search(r"\d", candidate):
		return False
	if candidate.lower() in {"date", "particulars", "invoice", "inv"}:
		return False
	if re.search(r"(?i)\b(?:invoice|dated?|date|dt)\b", candidate):
		return False
	if re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", candidate):
		return False
	candidate_compact = re.sub(r"\s+", "", candidate.upper())
	if re.fullmatch(r"\d{64}", candidate_compact):
		return False
	if re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]", candidate_compact):
		return False
	return True


def clean_invoice_no(value: str) -> str:
	"""Normalize invoice identifier: remove words like 'invoice', 'inv', 'dated', surrounding noise."""
	if not value:
		return ""
	s = str(value).strip()
	# Remove common leading/trailing labels and words
	s = re.sub(r"(?i)^(?:invoice\s*(?:no\.?|number)?\s*[:\-\s]*|dated?\s*[:\-\s]*|date\s*[:\-\s]*|dt\s*[:\-\s]*)", "", s)
	s = re.sub(r"(?i)\b(?:invoice|dated?|date|dt)\b[:\s-]*", "", s)
	# Remove enclosing words like 'Invoice No: 123' and keep the core token
	s = re.sub(r"[\s\-:,]+$", "", s)
	s = re.sub(r"^[:\-\s,]+", "", s)
	s = s.strip()
	# Avoid returning long 64-digit IRN-like numbers
	compact = re.sub(r"\s+", "", s)
	if re.fullmatch(r"\d{64}", compact):
		return ""
	return s


def find_first(patterns: Sequence[str], text: str, flags: int = re.IGNORECASE) -> str:
	for pattern in patterns:
		match = re.search(pattern, text, flags)
		if match:
			for group in match.groups():
				if group:
					return group.strip(" :-")
			return match.group(0).strip(" :-")
	return ""


def infer_invoice_no_from_source_name(source_name: str) -> str:
	if not source_name:
		return ""
	base_part = source_name.split("::", 1)[-1]
	base_name = Path(base_part).name
	for pattern in [
		r"(?:invoice\s*(?:reference)?|bill|document|credit\s*note)\s*(?:no|number)\s*[_\- .]*([A-Z0-9\-/]{3,40})\b",
	]:
		m = re.search(pattern, base_name, re.IGNORECASE)
		if m:
			candidate = re.sub(r"\s+", " ", m.group(1)).strip(" .:-")
			if is_valid_invoice_no(candidate):
				return candidate
	return ""


def infer_narration_from_source_name(source_name: str) -> str:
	if not source_name:
		return ""
	base_part = source_name.split("::", 1)[-1]
	name = re.sub(r"[_\\/\-]+", " ", base_part)

	# Example: "22 To 30 Sep'25"
	range_match = re.search(
		r"(\d{1,2}\s*(?:to|through)\s*\d{1,2}\s*[A-Za-z]{3,9}\s*['’]?\d{2,4})",
		name,
		re.IGNORECASE,
	)
	if range_match:
		return f"BALIC COMM {range_match.group(1).strip()}"

	# Example: "for date 11-22" / "11-22"
	short_range = re.search(r"\b(\d{1,2}\s*[-/]\s*\d{1,2})\b", name)

	# Example: "Feb 2026", "DEC'25"
	month_match = re.search(
		r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*['’\-]?\s*\d{2,4})\b",
		name,
		re.IGNORECASE,
	)
	if month_match:
		month_text = re.sub(r"\s+", " ", month_match.group(1)).strip()
		if short_range:
			return f"Brokerage Commission for the Month of {month_text} ({short_range.group(1).replace(' ', '')})"
		return f"Brokerage Commission for the Month of {month_text}"

	return ""


def words_to_number(words: str) -> str:
	"""Convert English words to numeric amount (e.g., 'Six thousand seven hundred' -> '6700').
	Handles paise like 'and paise seventy eight' where multiple words follow paise.
	"""
	if not words:
		return ""
	
	words = words.lower().strip()
	# Basic word-to-number mapping
	ones = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, 
	        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
	        "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
	        "eighteen": 18, "nineteen": 19}
	tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
	       "eighty": 80, "ninety": 90}
	scales = {"thousand": 1000, "lakh": 100000, "crore": 10000000}
	
	# Extract decimal part (paise) if present
	decimal_part = ""
	if "paise" in words:
		# Handle phrase like "... rupees seventy six paise".
		rupees_paise_match = re.search(r"rupees\s+(.+?)\s+paise", words)
		if rupees_paise_match:
			paise_words = rupees_paise_match.group(1).strip()
			paise_value = 0
			for paise_word in paise_words.split():
				if paise_word in ones:
					paise_value += ones[paise_word]
				elif paise_word in tens:
					paise_value += tens[paise_word]
			if paise_value > 0:
				decimal_part = f".{paise_value:02d}"
			words = words[:rupees_paise_match.start()]

		# First try numeric paise: "12 paise"
		paise_numeric = re.search(r"(\d+)\s+paise", words)
		if paise_numeric:
			paise_num = int(paise_numeric.group(1))
			decimal_part = f".{paise_num:02d}"
			words = words[:paise_numeric.start()]
		else:
			# Handle word-based paise: "paise seventy eight" or "and paise seventy eight"
			# Extract all words after paise until "only" or end
			paise_match = re.search(r"(?:and\s+)?paise\s+(.+?)(?:\s+only)?\s*$", words)
			if paise_match:
				paise_words = paise_match.group(1).strip()
				# Convert multi-word paise (e.g., "seventy eight" -> 78)
				paise_value = 0
				paise_current = 0
				for paise_word in paise_words.split():
					if paise_word in ones:
						paise_current += ones[paise_word]
					elif paise_word in tens:
						paise_current += tens[paise_word]
					elif paise_word.isdigit():
						paise_current = int(paise_word)
					elif paise_word not in ["and", "only"]:
						num_match = re.search(r"\d+", paise_word)
						if num_match:
							paise_current = int(num_match.group())
				paise_value = paise_current
				if paise_value > 0:
					decimal_part = f".{paise_value:02d}"
				words = words[:paise_match.start()]
	
	# Parse the main number with Indian grouping semantics.
	result = 0
	current_group = 0

	for word in words.split():
		if word in ones:
			current_group += ones[word]
		elif word in tens:
			current_group += tens[word]
		elif word == "hundred":
			if current_group == 0:
				current_group = 1
			current_group *= 100
		elif word in scales:
			scale = scales[word]
			if current_group == 0:
				continue
			result += current_group * scale
			current_group = 0
		elif word not in ["and", "only", "rupees"]:
			num_match = re.search(r"\d+", word)
			if num_match:
				current_group += int(num_match.group())

	result += current_group
	
	if result == 0 and not decimal_part:
		return ""
	
	# Format the result with decimal part
	if decimal_part:
		return f"{result}{decimal_part}" if result > 0 else f"0{decimal_part}"
	return str(result) if result > 0 else ""


def extract_tax_amount(text: str, tax_name: str) -> str:
	"""Extract tax amount (not percentage) from text like 'CGST 9% 511.02'"""
	if not text or not tax_name:
		return ""
	
	# Replace newlines with spaces to handle multi-line table format
	processed = text.replace('\n', ' ')

	# 1) Direct pattern: TAX ... RATE% AMOUNT
	pattern = rf"{tax_name}[^%]{{0,220}}?(\d{{1,2}}(?:\.\d+)?)\s*%[^0-9]{{0,12}}([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)"
	match = re.search(pattern, processed, re.IGNORECASE | re.DOTALL)
	if match:
		candidate = clean_amount(match.group(2))
		if candidate and float(candidate) > 30:
			return candidate

	# 2) Segment scan: find % AMOUNT pairs near tax name
	tax_match = re.search(rf"{tax_name}", processed, re.IGNORECASE)
	if tax_match:
		start = tax_match.start()
		search_text = processed[start:start + 420]
		pairs = re.findall(r"\d{1,2}(?:\.\d+)?\s*%[^0-9]{0,12}([0-9][0-9,]*(?:\.[0-9]{1,2})?)", search_text, re.IGNORECASE)
		for amount in pairs:
			candidate = clean_amount(amount)
			if candidate and float(candidate) > 30:
				return candidate

	# 3) Explicit label fallback: CGST amount 123.45
	label_match = re.search(rf"{tax_name}\s*(?:amount)?\s*[:\-]?\s*([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)(?!\s*%)", processed, re.IGNORECASE)
	if label_match:
		candidate = clean_amount(label_match.group(1))
		if candidate and float(candidate) > 30:
			return candidate
	
	return ""


def extract_tax_bundle_from_commission_line(text: str) -> Tuple[str, str, str, str]:
	"""Extract taxable, CGST, SGST, total from common line patterns."""
	processed = re.sub(r"\s+", " ", text)
	if "commission" not in processed.lower():
		return "", "", "", ""

	# Pattern A: taxable 9% cgst 9% sgst total
	pattern = r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+\d{1,2}(?:\.\d+)?\s*%\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+\d{1,2}(?:\.\d+)?\s*%\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)"
	match = re.search(pattern, processed, re.IGNORECASE)
	if match:
		taxable = clean_amount(match.group(1))
		cgst = clean_amount(match.group(2))
		sgst = clean_amount(match.group(3))
		total = clean_amount(match.group(4))
		return taxable, cgst, sgst, total

	# Pattern B: only repeated % amount pairs near commission (no explicit taxable/total)
	pairs = re.findall(r"\d{1,2}(?:\.\d+)?\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", processed, re.IGNORECASE)
	clean_pairs = [clean_amount(p) for p in pairs if clean_amount(p)]
	clean_pairs = [p for p in clean_pairs if float(p) > 10]
	if len(clean_pairs) >= 2:
		return "", clean_pairs[0], clean_pairs[1], ""

	return "", "", "", ""


def extract_generic_tax_amount_row(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/gst_total from compact five-number rows."""
	processed = re.sub(r"\s+", " ", text)
	patterns = [
		r"(?:amount\s+in|amount)\s*[:\-]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"hsn/sac[^0-9]{0,40}(?:amount\s+in)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	]
	for pattern in patterns:
		match = re.search(pattern, processed, re.IGNORECASE)
		if not match:
			continue
		vals = [clean_amount(match.group(i)) for i in range(1, 6)]
		try:
			nums = [float(v) for v in vals if v]
		except (ValueError, TypeError):
			continue
		if len(nums) == 5 and nums[0] > 100 and nums[1] > 0 and nums[2] > 0:
			return vals[0], vals[1], vals[2], vals[3], vals[4]
	return "", "", "", "", ""


def extract_credit_note_tax_row(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from Credit Note tax table rows."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "credit note" not in low:
		return "", "", "", "", ""

	segment_match = re.search(
		r"taxable\s*value.*?(?:total\s*invoice\s*value|amount\s*in\s*words|authorised\s*signatory|$)",
		processed,
		re.IGNORECASE,
	)
	if not segment_match:
		return "", "", "", "", ""

	segment = segment_match.group(0)
	nums = [clean_amount(n) for n in re.findall(r"[0-9][0-9,]*(?:\.[0-9]{1,2})?", segment)]
	vals: List[float] = []
	for n in nums:
		if not n:
			continue
		try:
			vals.append(float(n.replace(",", "")))
		except (ValueError, TypeError):
			continue

	# Sliding window pattern: taxable, cgst_rate, cgst_amt, sgst_rate, sgst_amt, igst_rate, igst_amt, total.
	for i in range(0, max(0, len(vals) - 7)):
		taxable, cgst_rate, cgst_amt, sgst_rate, sgst_amt, igst_rate, igst_amt, total = vals[i:i + 8]
		if taxable <= 10 or total <= 10:
			continue
		if not (0 <= cgst_rate <= 30 and 0 <= sgst_rate <= 30 and 0 <= igst_rate <= 30):
			continue
		if (cgst_amt + sgst_amt + igst_amt) <= 0:
			continue
		expected_total = taxable + cgst_amt + sgst_amt + igst_amt
		if abs(total - expected_total) > max(2.0, expected_total * 0.02):
			continue
		return (
			f"{taxable:.2f}",
			f"{cgst_amt:.2f}",
			f"{sgst_amt:.2f}",
			f"{igst_amt:.2f}",
			f"{total:.2f}",
		)

	return "", "", "", "", ""


def extract_igst_bundle(text: str) -> Tuple[str, str, str]:
	"""Extract taxable, IGST, total from IGST-centric table layouts (Axis/DBS style)."""
	processed = re.sub(r"\s+", " ", text)
	patterns = [
		# DBS style: Taxable Value* IGST Rate IGST Amt Total Invoice Value ... taxable 18.00% igst total
		r"taxable\s*value\*?\s+igst\s*rate\s+igst\s*amt\s+total\s*invoice\s*value.*?([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+\d{1,2}(?:\.\d+)?%\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		# IDFC-style: Taxable Value IGST Rate IGST Amount ... taxable 18 igst (without percent sign).
		r"taxable\s*value\s+igst\s*rate\s+igst\s*amount.*?([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+\d{1,2}(?:\.\d+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		# Axis style: optional HSN code before taxable, then @18% tax amount
		r"taxable\s*value(?:\s*of\s*supply)?[^0-9]{0,140}(?:[0-9]{4,8}\s+)?([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*@?\s*\d{1,2}(?:\.\d+)?%\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	]
	for pattern in patterns:
		match = re.search(pattern, processed, re.IGNORECASE)
		if not match:
			continue
		taxable = clean_amount(match.group(1))
		igst = clean_amount(match.group(2))
		total = clean_amount(match.group(3)) if len(match.groups()) >= 3 else ""
		try:
			if taxable and igst and float(taxable) > 10 and float(igst) > 10:
				if not total:
					total_candidate = clean_amount(find_first([
						r"total\s*invoice\s*value[^0-9]{0,40}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
						r"invoice\s*total\s*amount[^0-9]{0,40}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
						r"total\s*amount[^0-9]{0,20}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
					], processed))
					if total_candidate:
						total = total_candidate
					else:
						total = f"{float(taxable) + float(igst):.2f}"
				return taxable, igst, total
		except (TypeError, ValueError):
			continue
	return "", "", ""


def extract_output_igst_tax_row(text: str) -> Tuple[str, str, str]:
	"""Extract taxable/IGST/total from 'OUTPUT IGST@18%' style invoice sections."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if (
		"output igst" not in low
		and "integrated tax" not in low
		and "integrated gst" not in low
		and "igst@" not in low
	):
		return "", "", ""

	igst = clean_amount(find_first([
		r"output\s*igst\s*@?\s*18\s*%?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"integrated\s*tax[^0-9]{0,30}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"integrated\s*gst[^0-9]{0,30}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"igst\s*@?\s*18\s*%?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	taxable = clean_amount(find_first([
		r"18\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+total",
		r"taxable\s*tax\s*amount\s*amount\s*rate\s*value[^0-9]{0,30}(?:[0-9][0-9,]*(?:\.[0-9]{1,2})?\s+){2}18\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	total = clean_amount(find_first([
		r"\btotal\b[^0-9]{0,12}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		r"total\s*invoice\s*value[^0-9]{0,20}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
	], processed))

	try:
		igst_num = float(igst.replace(",", "")) if igst else 0.0
		taxable_num = float(taxable.replace(",", "")) if taxable else 0.0
		total_num = float(total.replace(",", "")) if total else 0.0
		# Some OCR layouts surface IGST amount in taxable slot; normalize from total.
		if taxable_num > 0 and igst_num > 0 and total_num > 0 and taxable_num <= (igst_num * 1.1):
			taxable_num = max(0.0, total_num - igst_num)
		if taxable_num <= 0 and total_num > 0 and igst_num > 0 and total_num > igst_num:
			taxable_num = total_num - igst_num
		if total_num <= 0 and taxable_num > 0 and igst_num > 0:
			total_num = taxable_num + igst_num
		if taxable_num > 10 and igst_num > 10 and total_num > 10:
			return f"{taxable_num:.2f}", f"{igst_num:.2f}", f"{total_num:.2f}"
	except (ValueError, AttributeError):
		pass

	return "", "", ""


def extract_state_gst_hsn_row(text: str) -> Tuple[str, str, str, str]:
	"""Extract taxable/cgst/sgst/total from HSN rows with 9%+9% state taxes."""
	processed = re.sub(r"\s+", " ", text)
	low = processed.lower()
	if "997161" not in processed and "997119" not in processed:
		return "", "", "", ""
	if "insurance brokerage" not in low and "brokerage income" not in low and "commission" not in low:
		return "", "", "", ""

	patterns = [
		# Row with explicit total at end: ... 997161 <taxable> 9 <cgst> 9 <sgst> <total>
		r"9971\d{2}\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+9(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+9(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		# Row with total earlier, then serial/taxable/rates: ... 997161 <total> <sno> <taxable> 0 0 9 <cgst> 9 <sgst>
		r"9971\d{2}\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+\d+\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+0(?:\.0+)?\s+0(?:\.0+)?\s+9(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+9(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		# OCR variant without clear HSN delimiter: <taxable> 0 0 9 <cgst> 9 <sgst>
		r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+0(?:\.0+)?\s+0(?:\.0+)?\s+9(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+9(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		# Row without explicit total in digits: ... 997161 <taxable> 9 <cgst> 9 <sgst> Total Invoice Value ...
		r"9971\d{2}\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+9(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+9(?:\.0+)?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)(?=\s+(?:total\s*invoice\s*value|service\s*category|note|whether))",
	]

	for idx, pattern in enumerate(patterns):
		for m in re.finditer(pattern, processed, re.IGNORECASE):
			vals = [clean_amount(m.group(i)) for i in range(1, len(m.groups()) + 1)]
			try:
				nums = [float(v.replace(",", "")) for v in vals]
			except (ValueError, TypeError, AttributeError):
				continue

			if idx == 0:
				taxable_n, cgst_n, sgst_n, total_n = nums
			elif idx == 1:
				total_n, taxable_n, cgst_n, sgst_n = nums
			elif idx == 2:
				taxable_n, cgst_n, sgst_n = nums
				total_n = 0.0
			else:
				taxable_n, cgst_n, sgst_n = nums
				total_n = 0.0

			if taxable_n <= 10 or cgst_n <= 10 or sgst_n <= 10:
				continue
			if abs(cgst_n - sgst_n) > max(cgst_n, sgst_n) * 0.2:
				continue
			expected_total = taxable_n + cgst_n + sgst_n
			if total_n <= 10:
				total_n = expected_total
			if abs(total_n - expected_total) > max(3.0, expected_total * 0.03):
				continue
			return f"{taxable_n:.2f}", f"{cgst_n:.2f}", f"{sgst_n:.2f}", f"{total_n:.2f}"

	return "", "", "", ""


def extract_tax_summary_amount_row(text: str) -> Tuple[str, str, str, str, str]:
	"""Extract taxable/cgst/sgst/igst/total from summary rows like Tax'ble Amt ... Tot Inv. Amt."""
	processed = re.sub(r"\s+", " ", text)
	match = re.search(
		r"tax'?ble\s*amt\s+cgst\s*amt\s+sgst\s*amt\s+igst\s*amt.*?tot\s*inv\.?\s*amt\s+(.{20,260})",
		processed,
		re.IGNORECASE,
	)
	if not match:
		return "", "", "", "", ""

	segment = match.group(1)
	nums = [clean_amount(n) for n in re.findall(r"[0-9][0-9,]*(?:\.[0-9]{1,2})?", segment)]
	nums = [n for n in nums if n]
	if len(nums) < 5:
		return "", "", "", "", ""

	taxable, cgst, sgst, igst = nums[0], nums[1], nums[2], nums[3]
	# Keep totals from the summary row itself; ignore trailing timestamps/GSTIN digits.
	total_idx = 9 if len(nums) >= 10 else 4
	total = nums[total_idx]
	try:
		if float(taxable) > 10 and float(total) > 10:
			return taxable, cgst, sgst, igst, total
	except (ValueError, TypeError):
		pass
	return "", "", "", "", ""


def extract_motilal_igst_amounts(text: str) -> Tuple[str, str, str]:
	"""Extract taxable, IGST amount and total from Motilal's rate/amount table row."""
	processed = re.sub(r"\s+", " ", text)
	if "motilal oswal financial services" not in processed.lower():
		return "", "", ""
	try:
		# Reliable total from footer line.
		total_raw = find_first([
			r"total\s*amount\s*[:\-]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
			r"total\s*value\s*[:\-]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		], processed)
		total_num = float(clean_amount(total_raw).replace(",", "")) if total_raw else 0.0
		if total_num <= 0:
			return "", "", ""

		# Motilal interstate rows typically carry "18.0 0 <IGST Amount> <Total>".
		igst_match = re.search(
			r"18(?:\.0+)?\s+0\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+[0-9][0-9,]*(?:\.[0-9]{1,2})?",
			processed,
			re.IGNORECASE,
		)
		igst_num = float(clean_amount(igst_match.group(1)).replace(",", "")) if igst_match else 0.0
		if igst_num <= 0 and total_num > 0:
			# Fallback: pick the largest non-rate amount smaller than total.
			vals = [float(clean_amount(n).replace(",", "")) for n in re.findall(r"[0-9][0-9,]*(?:\.[0-9]{1,2})?", processed) if clean_amount(n)]
			candidates = [v for v in vals if 0 < v < total_num and not (v <= 30 and abs(v - round(v)) < 1e-6)]
			if candidates:
				igst_num = max(candidates)

		if igst_num <= 0:
			return "", "", ""

		taxable_num = total_num - igst_num
		if taxable_num <= 0:
			return "", "", ""

		taxable = f"{taxable_num:.2f}"
		igst = f"{igst_num:.2f}"
		total = f"{total_num:.2f}"
		return taxable, igst, total
	except (ValueError, AttributeError):
		pass
	return "", "", ""


def try_parse_date(raw: str) -> str:
	raw = (raw or "").strip()
	if not raw:
		return ""
	candidate = raw.replace(".", "/").replace("-", "/")
	match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", candidate)
	if match:
		candidate = match.group(1)
	else:
		month_match = re.search(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b", raw)
		if month_match:
			candidate = month_match.group(0)

	parsed_candidates: List[pd.Timestamp] = []
	for dayfirst in (True, False):
		parsed = pd.to_datetime(candidate, dayfirst=dayfirst, errors="coerce")
		if pd.notna(parsed):
			parsed_candidates.append(parsed)
	if not parsed_candidates:
		return ""
	valid_candidates = [p for p in parsed_candidates if 2025 <= p.year <= 2026]
	if not valid_candidates:
		return ""
	current_ts = pd.Timestamp.now().normalize()
	parsed_candidates = valid_candidates
	parsed_candidates.sort(key=lambda p: (p > current_ts, -p.value))
	return parsed_candidates[0].strftime("%d/%m/%Y")


def infer_plausible_date_from_text(text: str) -> str:
	"""Extract a plausible invoice date from file/folder names or OCR text."""
	raw = (text or "").strip()
	if not raw:
		return ""
	patterns = [
		r"(?<!\d)(\d{1,2})[._/-](\d{1,2})[._/-](2025|2026|\d{2,4})(?!\d)",
		r"(?<!\d)(\d{2})(\d{2})(25|26)(?!\d)",
		r"(?<!\d)(\d{2})(\d{2})(2025|2026)(?!\d)",
		r"(?<!\d)(2025|2026)(\d{2})(\d{2})(?!\d)",
	]
	for pattern in patterns:
		match = re.search(pattern, raw)
		if not match:
			continue
		groups = match.groups()
		candidate = ""
		if len(groups) == 3 and groups[0].isdigit() and groups[1].isdigit():
			first, second, third = groups
			if len(third) == 2:
				third = f"20{third}"
			candidate = f"{first}/{second}/{third}"
		elif len(groups) == 3 and groups[0] in {"2025", "2026"}:
			year, month, day = groups
			candidate = f"{day}/{month}/{year}"
		if candidate:
			parsed = try_parse_date(candidate)
			if parsed:
				return parsed
	return ""


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
	gray = image.convert("L")
	gray = ImageEnhance.Contrast(gray).enhance(1.8)
	gray = gray.filter(ImageFilter.MedianFilter(size=3))
	thresholded = gray.point(lambda p: 255 if p > 160 else 0)
	return thresholded


def rotate_if_horizontal(image: Image.Image) -> Image.Image:
	width, height = image.size
	if width > (height * 1.2):
		return image.rotate(90, expand=True)
	return image


def score_ocr_text(text: str) -> int:
	if not text:
		return 0
	# Prefer OCR output with more alphanumeric content.
	return len(re.findall(r"[A-Za-z0-9]", text))


def score_receipt_ocr_text(text: str) -> int:
	if not text:
		return 0
	base = score_ocr_text(text)
	bonus = 0
	if re.search(r"\b(invoice|tax\s+invoice|inv\s*no|vendor\s*inv|gstin|hsn|sac)\b", text, re.IGNORECASE):
		bonus += 30
	if re.search(r"\b(cgst|sgst|utgst|igst|taxable|total)\b", text, re.IGNORECASE):
		bonus += 30
	amount_hits = len(re.findall(r"\b[0-9][0-9,]{1,}(?:\.[0-9]{1,2})?\b", text))
	bonus += min(50, amount_hits * 2)
	return base + bonus


MIN_OCR_ALNUM_SCORE = 50


def get_rapidocr_engine():
	global RAPIDOCR_ENGINE
	global RAPIDOCR_INIT_FAILED
	if RapidOCR is None:
		return None
	if RAPIDOCR_INIT_FAILED:
		return None
	if RAPIDOCR_ENGINE is None:
		try:
			RAPIDOCR_ENGINE = RapidOCR()
		except Exception as exc:
			LOGGER.warning("RapidOCR initialization failed: %s", exc)
			RAPIDOCR_INIT_FAILED = True
			return None
	return RAPIDOCR_ENGINE


def get_easyocr_reader():
	global EASYOCR_READER
	global EASYOCR_INIT_FAILED
	if easyocr is None:
		return None
	if EASYOCR_INIT_FAILED:
		return None
	if EASYOCR_READER is None:
		try:
			EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
		except Exception as exc:
			LOGGER.warning("EasyOCR initialization failed: %s", exc)
			EASYOCR_INIT_FAILED = True
			return None
	return EASYOCR_READER


def ocr_image_with_easyocr(image: Image.Image) -> str:
	reader = get_easyocr_reader()
	if reader is None:
		return ""
	try:
		arr = np.array(image)
		result = reader.readtext(arr, detail=0, paragraph=True)
		if not result:
			return ""
		return normalize_text("\n".join(str(x) for x in result if str(x).strip()))
	except Exception as exc:
		LOGGER.debug("EasyOCR failed for image: %s", exc)
		return ""


def ocr_image_with_tesseract(image: Image.Image) -> str:
	if pytesseract is None:
		return ""
	try:
		# OEM 1 + PSM 6 works well for dense invoice/table blocks.
		txt = pytesseract.image_to_string(image, config="--oem 1 --psm 6")
		return normalize_text(txt or "")
	except Exception as exc:
		LOGGER.debug("Tesseract failed for image: %s", exc)
		return ""


def ocr_image_single(image: Image.Image, prefer_google: bool = False, force_google: bool = False) -> str:
	prepared = preprocess_image_for_ocr(image)
	rapid_text = ""
	engine = get_rapidocr_engine()
	if engine is not None:
		try:
			image_array = np.array(prepared)
			result, _ = engine(image_array)
			if result:
				chunks = [line[1] for line in result if len(line) > 1 and line[1]]
				rapid_text = normalize_text("\n".join(chunks))
		except Exception as exc:
			LOGGER.warning("RapidOCR failed for image: %s", exc)

	vision_text = ""
	should_try_vision = prefer_google or len(rapid_text.strip()) < 20
	if should_try_vision:
		vision_candidates: List[str] = []
		primary = ocr_image_with_google_vision(prepared)
		if primary:
			vision_candidates.append(primary)

		# For force_google sources (e.g., J&K/Finozone), try extra variants when OCR remains weak.
		if force_google and score_ocr_text(primary) < 30:
			raw_candidate = ocr_image_with_google_vision(image)
			if raw_candidate:
				vision_candidates.append(raw_candidate)

			alt = image.convert("L")
			alt = ImageEnhance.Contrast(alt).enhance(2.4)
			alt = alt.filter(ImageFilter.SHARPEN)
			alt = alt.point(lambda p: 255 if p > 145 else 0)
			alt_candidate = ocr_image_with_google_vision(alt)
			if alt_candidate:
				vision_candidates.append(alt_candidate)

		if vision_candidates:
			vision_text = max(vision_candidates, key=score_ocr_text)

	if force_google and vision_text:
		return vision_text

	# Open-source ensemble fallback when primary OCR is weak or cloud OCR is unavailable.
	open_source_candidates: List[str] = [rapid_text]
	if score_receipt_ocr_text(rapid_text) < 90:
		easy_text = ocr_image_with_easyocr(prepared)
		if easy_text:
			open_source_candidates.append(easy_text)
		tess_text = ocr_image_with_tesseract(prepared)
		if tess_text:
			open_source_candidates.append(tess_text)
		# Alternate thresholded variant can recover faint scans.
		alt = image.convert("L")
		alt = ImageEnhance.Contrast(alt).enhance(2.2)
		alt = alt.filter(ImageFilter.SHARPEN)
		alt = alt.point(lambda p: 255 if p > 145 else 0)
		easy_alt = ocr_image_with_easyocr(alt)
		if easy_alt:
			open_source_candidates.append(easy_alt)
		tess_alt = ocr_image_with_tesseract(alt)
		if tess_alt:
			open_source_candidates.append(tess_alt)

	best_open_source = max(open_source_candidates, key=score_receipt_ocr_text) if open_source_candidates else ""
	candidates = [best_open_source, vision_text]
	best = max(candidates, key=score_receipt_ocr_text)
	return best or ""


def ocr_image(image: Image.Image, prefer_google: bool = False, force_google: bool = False) -> str:
	base = rotate_if_horizontal(image)
	best_text = ""
	best_score = -1

	for angle in [0, 90, 180, 270]:
		candidate = base if angle == 0 else base.rotate(angle, expand=True)
		text = ocr_image_single(candidate, prefer_google=prefer_google, force_google=force_google)
		score = score_ocr_text(text)
		if score > best_score:
			best_score = score
			best_text = text
		# Early stop when OCR quality is clearly high.
		if score >= 120:
			break

	return normalize_text(best_text)


def ocr_image_with_google_vision(image: Image.Image) -> str:
	global GOOGLE_VISION_CALL_COUNT, GOOGLE_VISION_DISABLED_FOR_RUN, GOOGLE_VISION_API_KEY
	api_key = get_google_vision_api_key()
	if not api_key:
		return ""

	try:
		buffer = io.BytesIO()
		image.save(buffer, format="PNG")
		content = base64.b64encode(buffer.getvalue()).decode("ascii")
		payload = {
			"requests": [
				{
					"image": {"content": content},
					"features": [{"type": "TEXT_DETECTION"}],
				}
			]
		}
		request = urllib.request.Request(
			url=f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
			data=json.dumps(payload).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		GOOGLE_VISION_CALL_COUNT += 1
		with urllib.request.urlopen(request, timeout=30) as response:
			body = response.read().decode("utf-8")
		parsed = json.loads(body)
		responses = parsed.get("responses", [])
		if not responses:
			return ""
		first = responses[0]
		full_text = first.get("fullTextAnnotation", {}).get("text", "")
		if not full_text:
			annotations = first.get("textAnnotations", [])
			if annotations:
				full_text = annotations[0].get("description", "")
		return normalize_text(full_text)
	except urllib.error.HTTPError as exc:
		body_text = ""
		try:
			body_text = exc.read().decode("utf-8", errors="replace")
		except Exception:
			body_text = ""
		if exc.code == 403:
			GOOGLE_VISION_DISABLED_FOR_RUN = True
			LOGGER.warning("Google Vision OCR disabled for this run after HTTP 403: %s", body_text or exc)
		else:
			LOGGER.warning("Google Vision OCR failed with HTTP %s: %s", exc.code, body_text or exc)
		return ""
	except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
		LOGGER.warning("Google Vision OCR failed: %s", exc)
		return ""


def probe_google_vision_api(image: Image.Image) -> Dict[str, str]:
	"""Run the Google Vision request once and return a structured diagnostic payload.

	This is useful for distinguishing API/auth problems from OCR parsing issues.
	"""
	api_key = get_google_vision_api_key()
	result: Dict[str, str] = {"ok": "false", "status": "no_api_key", "text": "", "error": ""}
	if not api_key:
		result["error"] = "GOOGLE_VISION_API_KEY is not configured"
		return result

	try:
		buffer = io.BytesIO()
		image.save(buffer, format="PNG")
		content = base64.b64encode(buffer.getvalue()).decode("ascii")
		payload = {
			"requests": [
				{
					"image": {"content": content},
					"features": [{"type": "TEXT_DETECTION"}],
				}
			]
		}
		request = urllib.request.Request(
			url=f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
			data=json.dumps(payload).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(request, timeout=30) as response:
			body = response.read().decode("utf-8")
		parsed = json.loads(body)
		responses = parsed.get("responses", [])
		first = responses[0] if responses else {}
		full_text = first.get("fullTextAnnotation", {}).get("text", "") or ""
		if not full_text:
			annotations = first.get("textAnnotations", [])
			if annotations:
				full_text = annotations[0].get("description", "") or ""
		result["ok"] = "true"
		result["status"] = "200"
		result["text"] = normalize_text(full_text)
		return result
	except urllib.error.HTTPError as exc:
		try:
			body_text = exc.read().decode("utf-8", errors="replace")
		except Exception:
			body_text = ""
		result["status"] = str(exc.code)
		result["error"] = body_text or str(exc)
		return result
	except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
		result["status"] = "error"
		result["error"] = str(exc)
		return result


def infer_password_from_name(file_name: str) -> Optional[str]:
	lower_name = file_name.lower()
	for key, password in BANK_PASSWORDS.items():
		if key in lower_name:
			return password
	return None


def infer_password_from_path(path: Path) -> Optional[str]:
	# Match password keys from any part of the path (folder names or filename).
	return infer_password_from_name(str(path))


def is_jk_bank_source(path: Path) -> bool:
	lower = str(path).lower()
	return (
		"jammu and kashmir bank" in lower
		or "j&k bank" in lower
		or "j k bank" in lower
	)


def is_monetary_ocr_priority_source(path: Path) -> bool:
	"""Sources where OCR significantly improves Brokerage/GST/Total extraction."""
	lower = str(path).lower()
	return (
		"dhanlaxmi bank" in lower
		or "nkgsb" in lower
		or "catalyst insurance" in lower
		or "beacon insurance" in lower
	)


def is_india_post_source(path: Path) -> bool:
	return "india post payments bank" in str(path).lower()


def get_poppler_path() -> Optional[str]:
	# Prefer bundled local poppler on Windows workspaces.
	candidates = [
		Path("poppler-25.12.0") / "Library" / "bin",
		Path("poppler") / "Library" / "bin",
		Path("poppler") / "bin",
	]
	for candidate in candidates:
		if candidate.exists() and candidate.is_dir():
			return str(candidate)
	return None


def render_pdf_page_image(pdf_bytes: bytes, page_num: int, dpi: int = 300) -> Optional[Image.Image]:
	"""Render one PDF page to PIL image.
	Prefers PyMuPDF (no Poppler dependency), falls back to pdf2image.
	"""
	if page_num <= 0:
		return None

	# 1) Poppler-free path via PyMuPDF.
	if fitz is not None:
		try:
			doc = fitz.open(stream=pdf_bytes, filetype="pdf")
			idx = page_num - 1
			if idx < 0 or idx >= len(doc):
				doc.close()
				return None
			page = doc.load_page(idx)
			scale = dpi / 72.0
			mat = fitz.Matrix(scale, scale)
			pix = page.get_pixmap(matrix=mat, alpha=False)
			mode = "RGB" if pix.n < 4 else "RGBA"
			img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
			doc.close()
			if mode == "RGBA":
				img = img.convert("RGB")
			return img
		except Exception as exc:
			LOGGER.debug("PyMuPDF render failed on page %s: %s", page_num, exc)

	# 2) Existing fallback via pdf2image (requires Poppler).
	if convert_from_bytes is None:
		return None
	try:
		poppler_path = get_poppler_path()
		if poppler_path:
			images = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path=poppler_path, first_page=page_num, last_page=page_num)
		else:
			images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=page_num, last_page=page_num)
		return images[0] if images else None
	except Exception as exc:
		LOGGER.debug("pdf2image render failed on page %s: %s", page_num, exc)
		return None


def pdf_page_texts(pdf_bytes: bytes, source_name: str, override_password: Optional[str]) -> List[Tuple[int, str]]:
	if PdfReader is None:
		raise RuntimeError("Missing dependency: pypdf")

	reader = PdfReader(io.BytesIO(pdf_bytes))
	if reader.is_encrypted:
		# Some PDFs are flagged as encrypted but are openable with an empty password.
		passwords_to_try: List[str] = [""]
		if override_password:
			if override_password not in passwords_to_try:
				passwords_to_try.append(override_password)
		guessed = infer_password_from_name(source_name)
		if guessed and guessed not in passwords_to_try:
			passwords_to_try.append(guessed)
		for password in BANK_PASSWORDS.values():
			if password not in passwords_to_try:
				passwords_to_try.append(password)

		opened = False
		for candidate in passwords_to_try:
			try:
				result = reader.decrypt(candidate)
				if result:
					opened = True
					LOGGER.info("Decrypted protected PDF %s", source_name)
					break
			except Exception:
				continue
		if not opened:
			raise ValueError(f"Unable to decrypt PDF: {source_name}")

	page_texts: List[Tuple[int, str]] = []
	for i, page in enumerate(reader.pages, start=1):
		raw_text = page.extract_text() or ""
		text = normalize_text(raw_text)
		page_texts.append((i, text))
	return page_texts


def pdf_ocr_page_texts(
	pdf_bytes: bytes,
	prefer_google: bool = False,
	force_google: bool = False,
	source_id: str = "",
	pdf_digest: str = "",
	page_numbers: Optional[Sequence[int]] = None,
) -> List[Tuple[int, str]]:
	if convert_from_bytes is None and fitz is None:
		return []

	# Determine page set to process.
	if page_numbers:
		requested_pages = sorted({int(p) for p in page_numbers if int(p) > 0})
	else:
		# Need page count when caller asks for all pages.
		try:
			if PdfReader is not None:
				requested_pages = list(range(1, len(PdfReader(io.BytesIO(pdf_bytes)).pages) + 1))
			else:
				requested_pages = []
		except Exception:
			requested_pages = []

	if not requested_pages:
		return []

	if not pdf_digest:
		pdf_digest = hashlib.sha1(pdf_bytes).hexdigest()

	page_text_map: Dict[int, str] = {}
	miss_pages: List[int] = []
	cache_hits = 0

	for p in requested_pages:
		ck = _ocr_cache_key(source_id, pdf_digest, p, prefer_google, force_google)
		cached = _ocr_cache_get(ck)
		if cached:
			page_text_map[p] = cached
			cache_hits += 1
		else:
			miss_pages.append(p)

	for p in miss_pages:
		image = render_pdf_page_image(pdf_bytes, p, dpi=300)
		if image is None:
			LOGGER.warning("PDF to image conversion failed on page %s", p)
			continue
		text = ocr_image(image, prefer_google=prefer_google, force_google=force_google)
		page_text_map[p] = text
		ck = _ocr_cache_key(source_id, pdf_digest, p, prefer_google, force_google)
		_ocr_cache_put(ck, text)

	if cache_hits:
		LOGGER.debug("OCR cache hit: %s/%s pages for %s", cache_hits, len(requested_pages), source_id or "<unknown>")

	return [(p, page_text_map.get(p, "")) for p in requested_pages]


def extract_zip_to_temp(zip_path: Path) -> Path:
	temp_dir = Path(tempfile.mkdtemp(prefix="receipt_zip_"))
	with ZipFile(zip_path, "r") as zip_ref:
		zip_ref.extractall(temp_dir)
	return temp_dir


def _excel_engine_for_path(path: Path) -> Optional[str]:
	suffix = path.suffix.lower()
	if suffix == ".xlsb":
		return "pyxlsb"
	if suffix == ".xls":
		return "xlrd"
	return None


def _is_meaningful_excel_value(value: object) -> bool:
	if value is None or (isinstance(value, float) and pd.isna(value)):
		return False
	text = str(value).strip()
	return bool(text and text.lower() != "nan")


def _excel_row_to_fields(row: Dict[str, object], source_file: str, sheet_name: str, row_number: int) -> Optional[Dict[str, str]]:
	row_text = " ".join(str(value).strip() for value in row.values() if _is_meaningful_excel_value(value))
	if not row_text:
		return None
	fields = _ai_row_to_output_fields(row)
	fields = apply_party_overrides(fields, Path(source_file), row_text)
	fields = backfill_agent_pan(fields, row_text)
	fields = normalize_mapping_anomalies(fields, row_text, Path(source_file))
	fields = enforce_tax_mode_fields(fields)
	fields = apply_gst_autofill(fields)
	if not (fields.get("Narration", "") or "").strip():
		fields["Narration"] = "COMMISSION"
	if not is_actual_receipt_row(fields, row_text):
		return None
	fields["Source File"] = source_file
	fields["Source Page"] = f"{sheet_name}:{row_number}"
	is_math_valid, math_reason = validate_math_extraction(fields)
	fields["Math Valid"] = "YES" if is_math_valid else f"NO: {math_reason}"
	fields["Missing Field and Why"] = build_missing_field_reason(fields)
	return fields


def process_spreadsheet(path: Path, source_hint: Optional[Path] = None, source_display: Optional[str] = None) -> List[ReceiptLineItem]:
	rows: List[ReceiptLineItem] = []
	effective_source = source_hint or path
	effective_display = source_display or str(path)
	engine = _excel_engine_for_path(path)
	read_kwargs = {"sheet_name": None, "dtype": object}
	if engine:
		read_kwargs["engine"] = engine
	try:
		workbook = pd.read_excel(path, **read_kwargs)
	except Exception as exc:
		LOGGER.warning("Failed to read spreadsheet %s: %s", path, exc)
		return rows

	if isinstance(workbook, pd.DataFrame):
		workbook = {"Sheet1": workbook}

	for sheet_name, sheet_df in workbook.items():
		if sheet_df is None or sheet_df.empty:
			continue
		for row_number, row in sheet_df.iterrows():
			row_dict = {str(col): row[col] for col in sheet_df.columns if _is_meaningful_excel_value(row.get(col))}
			if not row_dict:
				continue
			fields = _excel_row_to_fields(row_dict, str(effective_display), str(sheet_name), int(row_number) + 2)
			if not fields:
				continue
			rows.append(ReceiptLineItem(values=fields))

	return rows


def find_candidate_files(input_path: Path) -> Iterable[Path]:
	supported = {".pdf", ".jpg", ".jpeg", ".png", ".zip", ".xlsx", ".xlsm", ".xlsb", ".xls"}
	# If a single file is provided, yield it when supported.
	if input_path.is_file():
		if input_path.suffix.lower() in supported:
			LOGGER.debug("Found supported file: %s", input_path)
			yield input_path
		else:
			LOGGER.debug("Skipping unsupported input file: %s", input_path)
		return

	found = []
	skipped = []
	for path in input_path.rglob("*"):
		if path.is_file():
			sfx = path.suffix.lower()
			if sfx in supported:
				found.append(path)
			else:
				skipped.append(path)

	# Log summary for visibility so files aren't silently skipped.
	if found:
		LOGGER.info("Discovered %d candidate files under %s", len(found), input_path)
		for f in found:
			LOGGER.debug("Found: %s", f)
	else:
		LOGGER.warning("No supported files found under %s", input_path)

	if skipped:
		LOGGER.debug("Skipped %d unsupported files (showing up to 10): %s", len(skipped), skipped[:10])

	for p in found:
		yield p


def extract_text_from_image_file(path: Path) -> List[Tuple[int, str]]:
	image = Image.open(path)
	low = str(path).lower()
	force_google = "finozone" in low
	return [(1, ocr_image(image, prefer_google=True, force_google=force_google))]


def split_receipts_from_page_text(page_text: str) -> List[str]:
	separators = [
		r"\n\s*(?:invoice|tax\s+invoice)\s*(?:no|number)?\b",
		r"\n\s*vendor\s+inv\s+no\b",
		r"\n\s*irn\s*(?:no|number)?\b",
	]
	chunks = [page_text]
	for separator in separators:
		new_chunks: List[str] = []
		for chunk in chunks:
			parts = re.split(separator, chunk, flags=re.IGNORECASE)
			if len(parts) <= 1:
				new_chunks.append(chunk)
			else:
				for i, part in enumerate(parts):
					if i == 0:
						# Text before first invoice-like separator is page header context, not a receipt.
						continue
					prefixed = "Invoice No " + part
					if prefixed.strip():
						new_chunks.append(prefixed)
		chunks = new_chunks

	filtered: List[str] = []
	for chunk in chunks:
		candidate = chunk.strip()
		if len(candidate) <= 30:
			continue
		try:
			fields = extract_fields(candidate)
		except Exception:
			fields = {}
		has_inv_label = bool(re.search(r"\b(invoice\s*(?:no\.?|number|reference\s*no)|bill\s*(?:no\.?|number)|document\s*no\.?)\b", candidate, re.IGNORECASE))
		signals = sum(1 for k in ["Vendor Inv No", "Vendor Inv Date", "Total Inv Amt", "BROKERAGE Amount", "GST TOTAL AMT"] if (fields.get(k, "") or "").strip())
		if signals >= 2 and (is_valid_invoice_no(fields.get("Vendor Inv No", "")) or has_inv_label):
			filtered.append(candidate)
	return filtered if filtered else [page_text]


def validate_math_extraction(row: Dict[str, str]) -> Tuple[bool, str]:
	"""
	Validate extraction with strict commission math when brokerage is available.
	Rules:
	- GST mode must be clearly IGST-only or CGST+SGST/UTGST-only.
	- GST should be ~18% of brokerage.
	- Only brokerage and tax fields are used in this validation.
	Returns (is_valid, reason_for_mismatch).
	"""
	try:
		row_copy = dict(row)
		brok = _amount_to_float(row_copy.get("BROKERAGE Amount", ""))
		cgst = _amount_to_float(row_copy.get("CGST @ 9%", ""))
		sgst = _amount_to_float(row_copy.get("SGST @ 9%", ""))
		utgst = _amount_to_float(row_copy.get("UTGST", ""))
		igst = _amount_to_float(row_copy.get("IGST", ""))
		gst_total = _amount_to_float(row_copy.get("GST TOTAL AMT", ""))

		if brok <= 0 and gst_total <= 0 and cgst <= 0 and sgst <= 0 and utgst <= 0 and igst <= 0:
			return (False, "Missing monetary values")
		if brok <= 0:
			return (False, "BROKERAGE Amount missing or invalid")

		mode = _detect_clear_tax_mode(row_copy)
		if mode == "none":
			if gst_total > 0:
				return (False, f"Unable to verify GST mode; GST TOTAL={gst_total:.2f}, {_tax_mode_summary(row_copy)}")
			return (False, f"Unable to verify GST mode; {_tax_mode_summary(row_copy)}")

		if mode == "igst":
			if igst <= 0:
				return (False, f"IGST-only mode expected, but IGST is missing; {_tax_mode_summary(row_copy)}")
			effective_gst = igst
		else:
			# CGST + SGST/UTGST mode
			state_component = sgst if sgst > 0 else utgst
			if cgst <= 0 or state_component <= 0:
				return (False, f"CGST + SGST/UTGST mode expected, but one component is missing; {_tax_mode_summary(row_copy)}")
			
			# Calculate effective GST as sum of components
			effective_gst = cgst + state_component
			
			# Verify only that the two components are roughly balanced; avoid enforcing a fixed 9% assumption.
			if abs(cgst - state_component) > max(2.0, max(cgst, state_component) * 0.20):
				delta = abs(cgst - state_component)
				return (False, f"CGST {cgst:.2f} and SGST/UTGST {state_component:.2f} are not reasonably balanced by {delta:.2f}")

		expected_gst = brok * 0.18
		gst_tol = max(2.0, expected_gst * 0.05)
		if abs(effective_gst - expected_gst) > gst_tol:
			delta = abs(effective_gst - expected_gst)
			return (False, f"GST {effective_gst:.2f} differs from 18% of Brokerage {expected_gst:.2f} by {delta:.2f}")

		# Verify GST TOTAL matches the sum of components when it is present.
		if gst_total > 0 and abs(gst_total - effective_gst) > max(2.0, effective_gst * 0.02):
			delta = abs(gst_total - effective_gst)
			return (False, f"GST TOTAL {gst_total:.2f} differs from tax components {effective_gst:.2f} by {delta:.2f}")

		# Intentionally do not enforce a hard 18% brokerage-to-GST ratio here.

		return (True, "Math valid")
		
	except (ValueError, TypeError, AttributeError):
		return (False, "Could not parse amounts for validation")


def extract_fields(text: str, is_axis_bank: bool = False) -> Dict[str, str]:
	flattened = text.replace("\n", " ")
	flattened = re.sub(r"\s+", " ", flattened).strip()
	# Resolve GSTIN roles early so later PAN/state fallbacks can use them safely.
	balic_gstn, broker_gstn = choose_balic_and_broker_gstin(flattened)
	# OCR on scanned PDFs can distort the word "invoice" (e.g., lnvoice/lNVOICE).
	invoice_search_text = flattened
	invoice_search_text = re.sub(r"\b[il1|]nvoice\b", "invoice", invoice_search_text, flags=re.IGNORECASE)
	invoice_search_text = re.sub(r"\binv[o0]ice\b", "invoice", invoice_search_text, flags=re.IGNORECASE)
	invoice_search_text = re.sub(r"\binvo1ce\b", "invoice", invoice_search_text, flags=re.IGNORECASE)
	
	invoice_patterns = [
		r"\b(KI\d{8,})\b",	# Incred invoice numbers
		r"\b(FIN\d{10,})\b",
		r"e\s*[- ]?invoice\s*(?:no|number)\s*[:\-]?\s*([A-Z0-9\-/]{6,60})",
		r"(?:si\.?\s*no\.?\s*)?invoice\s*no\.?\s+invoice\s*date\s+trans\.?\s*ref\.?\s*no\.?\s+charges.*?\b\d+\s+([A-Z0-9\-/]{8,40})\b",
		r"invoice\s*no\.?\s+invoice\s*date\s+([A-Z0-9\-/]{8,40})\s+[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4}",
		r"invoice\s*reference\s*no\.?\s*[:\-]?\s*([A-Z0-9\-/]{6,40})",
		r"bill\s*(?:no|number)\.?\s*[:\-]?\s*([A-Z0-9\-/]{6,40})",
		r"credit\s*note\s*reference\s*no\.?\s*[:\-]?\s*([A-Z0-9\-/]{6,40})",
		r"document\s*no\.?\s*[:\-]?\s*([A-Z0-9\-/]{6,40})",
		r"invoice\s*no\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/ ]{5,50})(?:\s+(?:date|dt)\b|\s+[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
		r"(?:Ref\s*Invoice\s*No|vendor\s*inv(?:oice)?\s*(?:no|number)|invoice\s*(?:no|number))\s*\.?\s*[:\-]?\s*([A-Z0-9\-/]+)",
		r"(?:invoice\s*no|ref\s*no)\.?\s*[:\-]?\s*(?:transaction\s+remarks\s+)?(?:[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4}\s+)?([A-Z]{3}\d{10,})",
		r"\b(inv\d{3,}[A-Z0-9\-/]*)\b",
	]
	invoice_no = ""
	for pattern in invoice_patterns:
		for match in re.finditer(pattern, invoice_search_text, re.IGNORECASE):
			candidate = ""
			if match.groups():
				for group in match.groups():
					if group:
						candidate = group.strip(" :-")
						break
			else:
				candidate = match.group(0).strip(" :-")
			candidate = re.sub(r"\s+", " ", candidate).strip(" .:-")
			candidate = re.sub(r"\s+(?:dated?|dt)\b.*$", "", candidate, flags=re.IGNORECASE).strip(" .:-")
			ctx_window = invoice_search_text[max(0, match.start() - 48): min(len(invoice_search_text), match.end() + 48)].lower()
			if re.search(r"\b(?:ack|acknowledg(?:e)?ment|ackn?\.?\s*no|irn|e\s*[- ]?way\s*bill)\b", ctx_window, re.IGNORECASE):
				continue
			if len(candidate) >= 2 and candidate[-2:].lower() in {"no", "dt"}:
				candidate = candidate[:-2].strip(" .:-")
			if re.search(r"\b(?:gstin|pan|cin|ifsc|hsn|sac)\b", candidate, re.IGNORECASE):
				continue
			if is_valid_invoice_no(candidate):
				invoice_no = candidate
				break
		if invoice_no:
			break

	if not invoice_no:
		label_iter = re.finditer(
			r"(?:invoice\s*(?:reference)?|bill|document)\s*(?:no|number|#)\.?\s*[:\-]?",
			invoice_search_text,
			re.IGNORECASE,
		)
		id_token_pattern = re.compile(
			r"\b(?:FIN\d{10,}|FZ\d{6,}|TCR\s*\d{2,}|[A-Z]{2,8}/\d{2,4}[-/]\d{2,4}/\d{1,6}|[A-Z0-9]{2,12}\d{3,}[A-Z0-9\-/]{0,30}|\d{6,}[A-Z][A-Z0-9\-/]{2,30})\b",
			re.IGNORECASE,
		)
		excluded_nearby = {"gstin", "pan", "cin", "ifsc", "hsn", "sac", "msme", "account", "a/c", "ack", "acknowledgement", "irn", "e-way"}
		for lm in label_iter:
			window = invoice_search_text[lm.end(): lm.end() + 180]
			for tm in id_token_pattern.finditer(window):
				cand = re.sub(r"\s+", " ", tm.group(0)).strip(" .:-")
				cand = re.sub(r"\s+(?:dated?|dt)\b.*$", "", cand, flags=re.IGNORECASE).strip(" .:-")
				near = window[max(0, tm.start() - 20): min(len(window), tm.end() + 20)].lower()
				if any(x in near for x in excluded_nearby):
					continue
				if re.search(r"\b(?:gstin|pan|cin|ifsc|hsn|sac)\b", cand, re.IGNORECASE):
					continue
				if cand.isdigit():
					# Short numeric tokens are usually serial/pin values unless very close to invoice label.
					if len(cand) < 10 and tm.start() > 40:
						continue
					if len(cand) == 6 and re.search(r"\b(?:pin|state|address|city)\b", near, re.IGNORECASE):
						continue
				if is_valid_invoice_no(cand):
					invoice_no = cand
					break
			if invoice_no:
				break

	# For dates, first try to find the label, then search for date pattern nearby
	# Handle both "invoice date" and "Ref Invoice Date" formats
	vendor_date_text = find_first(
		[
			r"(?:Ref\s*Invoice\s*Date|vendor\s*inv(?:oice)?\s*date|invoice\s*date|dated)\s*[:\-]?\s*([0-9]{1,2})[./-]([0-9]{1,2})[./-]([0-9]{2,4})",
		],
		flattened,
	)
	if vendor_date_text:
		# try_parse_date expects d/m/y format
		parts = re.search(r"([0-9]{1,2})[./-]([0-9]{1,2})[./-]([0-9]{2,4})", vendor_date_text)
		if parts:
			vendor_date = try_parse_date(f"{parts.group(1)}/{parts.group(2)}/{parts.group(3)}")
		else:
			vendor_date = try_parse_date(vendor_date_text)
	else:
		# Fallback: search for date patterns without requiring a label
		date_match = re.search(r"([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})", flattened)
		vendor_date = try_parse_date(date_match.group(1)) if date_match else ""

	amount_total = clean_amount(
		find_first(
			[
				r"total\s*amount\s*after\s*tax\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
				r"(?:total\s*inv(?:oice)?\s*amt|total\s*invoice\s*amount|total\s*invoice\s*value(?:\s*\([^)]*\))?|invoice\s*value(?:\s*\([^)]*\))?|total\s*amount\s*payable|total\s*amount)\s*(?:[:\-]|is)?\s*(?:rs\.?\s*)?([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.\d{1,2})?|[0-9]+(?:\.\d{1,2})?)",
				r"\btotal\s*(?:₹|rs\.?\s*)?([0-9][0-9,]*\.[0-9]{1,2})\b",
				r"(?:total\s*amount\s*payable|total\s*invoice\s*value)\s*[:\-]?\s*([0-9][0-9,]*)\b",
				r"(?:gross\s*amount|imf\s*fees)\s*[:\-]?\s*(\(?[0-9,]+(?:\.\d{1,2})?\)?)",
			],
			flattened,
		)
	)
	try:
		if amount_total and float(amount_total.replace(",", "")) <= 0:
			amount_total = ""
	except (ValueError, AttributeError):
		pass

	# IGST-first table fallback for Axis/DBS layouts
	igst_taxable, igst_bundle_amount, igst_bundle_total = extract_igst_bundle(flattened)
	if (not amount_total) and igst_bundle_total:
		amount_total = igst_bundle_total

	# OUTPUT IGST-style fallback (common in some broker invoices).
	out_taxable, out_igst, out_total = extract_output_igst_tax_row(flattened)
	if (not amount_total) and out_total:
		amount_total = out_total
	# Probitas label-driven extraction fallback.
	pb_taxable, pb_cgst, pb_sgst, pb_igst, pb_total = extract_probitas_amounts(flattened)
	out_igst_preferred = bool(re.search(r"integrated\s*(?:gst|tax)|output\s*igst", flattened, re.IGNORECASE))

	# IGST-only triplet fallback: <taxable> 18% <igst> <total>
	igst_triplet_taxable = ""
	igst_triplet_tax = ""
	igst_triplet_total = ""
	for m in re.finditer(
		r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+18\s*%\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		flattened,
		re.IGNORECASE,
	):
		taxable_s = clean_amount(m.group(1))
		igst_s = clean_amount(m.group(2))
		total_s = clean_amount(m.group(3))
		try:
			taxable_n = float(taxable_s) if taxable_s else 0.0
			igst_n = float(igst_s) if igst_s else 0.0
			total_n = float(total_s) if total_s else 0.0
			if taxable_n <= 10 or igst_n <= 10 or total_n <= 10:
				continue
			if abs(total_n - (taxable_n + igst_n)) <= max(2.0, total_n * 0.03):
				igst_triplet_taxable = taxable_s
				igst_triplet_tax = igst_s
				igst_triplet_total = total_s
				break
		except (ValueError, TypeError):
			continue

	# Prefer amount-in-words conversion when present to avoid OCR decimal corruption (e.g., 6700.04 -> 6704)
	amount_words_match = re.search(
		r"(?:Amount\s+in\s+words?|Amount\s+Chargeable\s*\(\s*in\s*words\s*\))\s*[:-]?\s*([A-Za-z\s]+?)(?:\s+only|\s+igst\b|\s+cgst\b|\s+sgst\b|\s+taxable\b|$)",
		flattened,
		re.IGNORECASE,
	)
	if amount_words_match:
		words_text = amount_words_match.group(1)
		converted = words_to_number(words_text)
		if converted and not amount_total:
			amount_total = converted

	# Tax amount in words (common in some scanned invoices, including Finozone layout)
	tax_words_match = re.search(
		r"Tax\s*Amount\s*\(\s*in\s*words\s*\)\s*[:\-]?\s*([A-Za-z\s]+?)(?:\s+only|\s+we\s+declare|\s+bank\s+details|\s+customer\b|$)",
		flattened,
		re.IGNORECASE,
	)
	tax_total_from_words = ""
	if tax_words_match:
		tax_total_from_words = words_to_number(tax_words_match.group(1))
	
	# Brokerage = Taxable Value (appears in the tax table)
	brokerage = clean_amount(
		find_first(
			[
				r"total\s*amount\s*before\s*tax\s*[:|]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
				r"(?:Taxable\s+Value|brokerage\s*amount|commission\s*amount)\s*[:\-]?\s*(\(?[0-9,]+(?:\.\d{1,2})?\)?)",
				r"\bbrokerage\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
			],
			flattened,
		)
	)
	if not brokerage and igst_taxable:
		brokerage = igst_taxable
	if not brokerage and out_taxable:
		brokerage = out_taxable
	if not brokerage and igst_triplet_taxable:
		brokerage = igst_triplet_taxable
	if not brokerage and pb_taxable:
		brokerage = pb_taxable
	if (not amount_total) and pb_total:
		amount_total = pb_total

	# Commission-line fallback (common receipt layout)
	bundle_taxable, bundle_cgst, bundle_sgst, bundle_total = extract_tax_bundle_from_commission_line(flattened)
	if not brokerage and bundle_taxable:
		brokerage = bundle_taxable
	if (not amount_total) and bundle_total:
		amount_total = bundle_total

	# NKGSB explicit amount row fallback: Amount taxable cgst sgst igst total
	nk_taxable, nk_cgst, nk_sgst, nk_igst, nk_total = extract_nkgsb_amount_row(flattened)
	company_specific_authoritative = False
	if not brokerage and nk_taxable:
		brokerage = nk_taxable
	if (not amount_total) and nk_total:
		amount_total = nk_total

	# Dhanlaxmi label-driven table extraction fallback.
	dh_taxable, dh_cgst, dh_sgst, dh_igst, dh_total = extract_dhanlaxmi_amounts(flattened)
	if "dhanlaxmi" in flattened.lower() and any([dh_taxable, dh_cgst, dh_sgst, dh_igst, dh_total]):
		company_specific_authoritative = True
		brokerage = dh_taxable or brokerage
		amount_total = dh_total or amount_total
	if not brokerage and dh_taxable:
		brokerage = dh_taxable
	if (not amount_total) and dh_total:
		amount_total = dh_total

	# City Union label-driven table extraction fallback.
	cu_taxable, cu_cgst, cu_sgst, cu_igst, cu_total = extract_city_union_amounts(flattened)
	if not brokerage and cu_taxable:
		brokerage = cu_taxable
	if (not amount_total) and cu_total:
		amount_total = cu_total

	# Ethika label-driven extraction fallback.
	et_taxable, et_cgst, et_sgst, et_igst, et_total = extract_ethika_amounts(flattened)
	if not brokerage and et_taxable:
		brokerage = et_taxable
	if (not amount_total) and et_total:
		amount_total = et_total

	# Catalyst OCR extraction fallback.
	ca_taxable, ca_cgst, ca_sgst, ca_igst, ca_total = extract_catalyst_amounts(flattened)
	if not brokerage and ca_taxable:
		brokerage = ca_taxable
	if (not amount_total) and ca_total:
		amount_total = ca_total

	# Incred amount extraction fallback.
	ic_taxable, ic_cgst, ic_sgst, ic_igst, ic_total = extract_incred_amounts(flattened)
	if not brokerage and ic_taxable:
		brokerage = ic_taxable
	if (not amount_total) and ic_total:
		amount_total = ic_total

	# JB Boda amount extraction fallback.
	jb_taxable, jb_cgst, jb_sgst, jb_igst, jb_total = extract_jb_boda_amounts(flattened)
	if not brokerage and jb_taxable:
		brokerage = jb_taxable
	if (not amount_total) and jb_total:
		amount_total = jb_total

	# Ideal Insurance table extraction fallback.
	id_taxable, id_cgst, id_sgst, id_igst, id_total = extract_ideal_amounts(flattened)
	if id_taxable and not company_specific_authoritative:
		brokerage = id_taxable
	if id_total and not company_specific_authoritative:
		amount_total = id_total

	# Mahindra Insurance table extraction fallback.
	mh_taxable, mh_cgst, mh_sgst, mh_igst, mh_total = extract_mahindra_amounts(flattened)
	if mh_taxable and not company_specific_authoritative:
		brokerage = mh_taxable
	if mh_total and not company_specific_authoritative:
		amount_total = mh_total

	# Bajaj Housing Finance table extraction fallback.
	bh_taxable, bh_cgst, bh_sgst, bh_igst, bh_total = extract_bajaj_housing_amounts(flattened)
	if bh_taxable and not company_specific_authoritative:
		brokerage = bh_taxable
	if bh_total and not company_specific_authoritative:
		amount_total = bh_total

	# Coverkraft table extraction fallback.
	ck_taxable, ck_cgst, ck_sgst, ck_igst, ck_total = extract_coverkraft_amounts(flattened)
	if ck_taxable and not company_specific_authoritative:
		brokerage = ck_taxable
	if ck_total and not company_specific_authoritative:
		amount_total = ck_total

	# HSN row fallback for common 9%+9% brokerage tables.
	hsn_taxable, hsn_cgst, hsn_sgst, hsn_total = extract_state_gst_hsn_row(flattened)
	if (not brokerage) and hsn_taxable:
		brokerage = hsn_taxable
	elif hsn_taxable and brokerage in {"997161", "997119", "997116"}:
		brokerage = hsn_taxable
	if (not amount_total) and hsn_total:
		amount_total = hsn_total

	# Generic five-number amount row fallback (common in OCR'd tax tables)
	gen_taxable, gen_cgst, gen_sgst, gen_igst, gen_gst_total = extract_generic_tax_amount_row(flattened)

	# Credit Note row fallback: Taxable Value / CGST / SGST / IGST / Total Value.
	cn_taxable, cn_cgst, cn_sgst, cn_igst, cn_total = extract_credit_note_tax_row(flattened)
	if (not brokerage) and cn_taxable:
		brokerage = cn_taxable
	if (not amount_total) and cn_total:
		amount_total = cn_total

	# Summary-row fallback: Tax'ble Amt / CGST Amt / SGST Amt / IGST Amt / Tot Inv. Amt
	sum_taxable, sum_cgst, sum_sgst, sum_igst, sum_total = extract_tax_summary_amount_row(flattened)
	if (not brokerage) and sum_taxable:
		brokerage = sum_taxable
	if (not amount_total) and sum_total:
		amount_total = sum_total

	# Motilal interstate layout fallback: IGST value appears in compact rate/amount table.
	mot_taxable, mot_igst, mot_total = extract_motilal_igst_amounts(flattened)
	if (not brokerage) and mot_taxable:
		brokerage = mot_taxable
	if (not amount_total) and mot_total:
		amount_total = mot_total

	# Jammu & Kashmir Bank OCR fallback: Taxable + GST amounts + Total Amount Payable
	jk_taxable, jk_cgst, jk_sgst, jk_igst, jk_total = extract_jk_bank_amounts(flattened)
	if not brokerage and jk_taxable:
		brokerage = jk_taxable
	if (not amount_total) and jk_total:
		amount_total = jk_total

	specific_taxable_hit = any([
		nk_taxable,
		dh_taxable,
		cu_taxable,
		pb_taxable,
		et_taxable,
		ca_taxable,
		ic_taxable,
		jb_taxable,
		id_taxable,
		mh_taxable,
		bh_taxable,
		ck_taxable,
		hsn_taxable,
		cn_taxable,
		sum_taxable,
		mot_taxable,
		jk_taxable,
		igst_taxable,
		out_taxable,
		igst_triplet_taxable,
	])
	specific_state_cgst = next((v for v in [dh_cgst, nk_cgst, cu_cgst, pb_cgst, et_cgst, ca_cgst, ic_cgst, jb_cgst, id_cgst, mh_cgst, bh_cgst, ck_cgst, hsn_cgst, cn_cgst, sum_cgst, bundle_cgst] if v), "")
	specific_state_sgst = next((v for v in [dh_sgst, nk_sgst, cu_sgst, pb_sgst, et_sgst, ca_sgst, ic_sgst, jb_sgst, id_sgst, mh_sgst, bh_sgst, ck_sgst, hsn_sgst, cn_sgst, sum_sgst, bundle_sgst] if v), "")
	if (specific_state_cgst or specific_state_sgst) and not company_specific_authoritative:
		company_specific_authoritative = True
	if (not brokerage) and (not specific_taxable_hit) and gen_taxable:
		brokerage = gen_taxable
	
	# Prefer IGST extraction first for Axis/DBS layouts and compact table rows (amount before rate).
	igst = ""
	if company_specific_authoritative:
		igst = dh_igst
	else:
		igst = clean_amount(find_first([
			r"\bigst\s*@?\s*\d{1,2}(?:\.\d+)?\s*%\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
		], flattened))
		if not igst:
			igst = clean_amount(extract_tax_amount(flattened, "IGST"))
		if not igst:
			igst = clean_amount(find_first([
				r"igst\s*(?:rate\s*)?(?:amt|amount)?\s*[:\-]?\s*([0-9,]+(?:\.\d{1,2})?)",
				r"\d{1,2}(?:\.\d+)?\s*%\s*([0-9,]+(?:\.\d{1,2})?)\s+(?:total\s*invoice\s*value|total\s*amount)",
			], flattened))
		if (not igst) and igst_bundle_amount:
			igst = igst_bundle_amount
		if (not igst) and out_igst:
			igst = out_igst
		if out_igst_preferred and out_igst:
			igst = out_igst
			if out_taxable:
				brokerage = out_taxable
			if out_total:
				amount_total = out_total
		if (not igst) and igst_triplet_tax:
			igst = igst_triplet_tax
		if (not igst) and nk_igst:
			igst = nk_igst
		if (not igst) and jk_igst:
			igst = jk_igst
		if (not igst) and cn_igst:
			igst = cn_igst
		if (not igst) and sum_igst:
			igst = sum_igst
		if (not igst) and mot_igst:
			igst = mot_igst
		if (not igst) and (not specific_taxable_hit) and gen_igst:
			igst = gen_igst
		if (not igst) and dh_igst:
			igst = dh_igst
		if (not igst) and cu_igst:
			igst = cu_igst
		if (not igst) and pb_igst:
			igst = pb_igst
		if (not igst) and et_igst:
			igst = et_igst
		if (not igst) and ca_igst:
			igst = ca_igst
		if (not igst) and ic_igst:
			igst = ic_igst
		if (not igst) and jb_igst:
			igst = jb_igst
		if (not igst) and id_igst:
			igst = id_igst
		if (not igst) and mh_igst:
			igst = mh_igst
		if (not igst) and bh_igst:
			igst = bh_igst
		if (not igst) and ck_igst:
			igst = ck_igst
	try:
		if igst and float(igst.replace(",", "")) <= 0:
			igst = ""
	except (ValueError, AttributeError):
		pass

	has_igst_layout = bool(re.search(
		r"igst\s*(?:rate|amt|amount)|igst\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?\s*%\s*\d{1,2}(?:\.[0-9]+)?|sgst\s*/\s*utgst\s+igst",
		flattened,
		re.IGNORECASE,
	))
	if igst_triplet_tax:
		has_igst_layout = True
	if out_igst:
		has_igst_layout = True

	if re.search(r"integrated\s*(?:gst|tax)|output\s*igst|igst\s*@?\s*18", flattened, re.IGNORECASE):
		has_igst_layout = True

	# Initialize state-tax fields; fill only when IGST is not dominant.
	cgst = ""
	sgst = ""
	utgst = ""

	try:
		igst_num = float(igst.replace(",", "")) if igst else 0
	except (ValueError, AttributeError):
		igst_num = 0

	if company_specific_authoritative:
		cgst = specific_state_cgst
		sgst = specific_state_sgst
		utgst = ""
		if igst and (cgst or sgst):
			igst = ""
	elif not (has_igst_layout and igst_num > 10):
		# Extract tax amounts using helper that focuses on amounts not percentages
		cgst = clean_amount(extract_tax_amount(flattened, "CGST"))
		if not cgst or float(cgst) <= 10:
			cgst = clean_amount(find_first([
				r"cgst\s*(?:@|at)?\s*\d{1,2}(?:\.\d+)?\s*%\s*([0-9,]+(?:\.\d{1,2})?)",
				r"cgst\s*(?:amount)?\s*[:\-]?\s*([0-9,]+(?:\.\d{1,2})?)",
			], flattened))

		sgst = clean_amount(extract_tax_amount(flattened, "SGST"))
		if not sgst or float(sgst) <= 10:
			sgst = clean_amount(find_first([
				r"sgst\s*(?:@|at)?\s*\d{1,2}(?:\.\d+)?\s*%\s*([0-9,]+(?:\.\d{1,2})?)",
				r"sgst\s*(?:amount)?\s*[:\-]?\s*([0-9,]+(?:\.\d{1,2})?)"
			], flattened))

		# If one side is missing, mirror from the other because these invoices carry both CGST and SGST
		if (not cgst or float(cgst) <= 10) and bundle_cgst and float(bundle_cgst) > 10:
			cgst = bundle_cgst
		if (not sgst or float(sgst) <= 10) and bundle_sgst and float(bundle_sgst) > 10:
			sgst = bundle_sgst
		if (not cgst or float(cgst) <= 10) and nk_cgst:
			cgst = nk_cgst
		if (not sgst or float(sgst) <= 10) and nk_sgst:
			sgst = nk_sgst
		if (not cgst or float(cgst) <= 10) and jk_cgst:
			cgst = jk_cgst
		if (not sgst or float(sgst) <= 10) and jk_sgst:
			sgst = jk_sgst
		if (not cgst or float(cgst) <= 10) and dh_cgst:
			cgst = dh_cgst
		if (not sgst or float(sgst) <= 10) and dh_sgst:
			sgst = dh_sgst
		if (not cgst or float(cgst) <= 10) and cu_cgst:
			cgst = cu_cgst
		if (not sgst or float(sgst) <= 10) and cu_sgst:
			sgst = cu_sgst
		if (not cgst or float(cgst) <= 10) and ic_cgst:
			cgst = ic_cgst
		if (not sgst or float(sgst) <= 10) and ic_sgst:
			sgst = ic_sgst
		if (not cgst or float(cgst) <= 10) and jb_cgst:
			cgst = jb_cgst
		if (not sgst or float(sgst) <= 10) and jb_sgst:
			sgst = jb_sgst
		if (not cgst or float(cgst) <= 10) and id_cgst:
			cgst = id_cgst
		if (not sgst or float(sgst) <= 10) and id_sgst:
			sgst = id_sgst
		if (not cgst or float(cgst) <= 10) and mh_cgst:
			cgst = mh_cgst
		if (not sgst or float(sgst) <= 10) and mh_sgst:
			sgst = mh_sgst
		if (not cgst or float(cgst) <= 10) and bh_cgst:
			cgst = bh_cgst
		if (not sgst or float(sgst) <= 10) and bh_sgst:
			sgst = bh_sgst
		if (not cgst or float(cgst) <= 10) and ck_cgst:
			cgst = ck_cgst
		if (not sgst or float(sgst) <= 10) and ck_sgst:
			sgst = ck_sgst
		if (not cgst or float(cgst) <= 10) and hsn_cgst:
			cgst = hsn_cgst
		if (not sgst or float(sgst) <= 10) and hsn_sgst:
			sgst = hsn_sgst
		if (not cgst or float(cgst) <= 10) and gen_cgst:
			cgst = gen_cgst
		if (not sgst or float(sgst) <= 10) and gen_sgst:
			sgst = gen_sgst
		if (not cgst or float(cgst) <= 10) and cn_cgst:
			cgst = cn_cgst
		if (not sgst or float(sgst) <= 10) and cn_sgst:
			sgst = cn_sgst
		if (not cgst or float(cgst) <= 10) and sum_cgst:
			cgst = sum_cgst
		if (not sgst or float(sgst) <= 10) and sum_sgst:
			sgst = sum_sgst

		# Final fallback: scan % amount pairs only around tax labels to avoid years/IDs.
		if (not cgst or float(cgst) <= 10) or (not sgst or float(sgst) <= 10):
			tax_segment_match = re.search(r"(?:cgst|sgst|utgst|taxable).*?(?:total\s+invoice\s+amount|total\s+amount|bank\s+details|declaration|$)", flattened, re.IGNORECASE)
			tax_segment = tax_segment_match.group(0) if tax_segment_match else ""
			global_pairs = re.findall(r"\d{1,2}(?:\.\d+)?\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", tax_segment, re.IGNORECASE)
			global_pairs = [clean_amount(p) for p in global_pairs if clean_amount(p)]
			global_pairs = [p for p in global_pairs if float(p) > 10]
			if len(global_pairs) >= 2:
				if (not cgst or float(cgst) <= 10):
					cgst = global_pairs[0]
				if (not sgst or float(sgst) <= 10):
					sgst = global_pairs[1]

		if (not cgst or float(cgst) <= 10) and sgst and float(sgst) > 10:
			cgst = sgst
		if (not sgst or float(sgst) <= 10) and cgst and float(cgst) > 10:
			sgst = cgst

		# Last-resort backfill when labels are weak in OCR text
		try:
			if (not cgst or float(cgst) <= 10) and (not sgst or float(sgst) <= 10):
				gst_total_hint = clean_amount(find_first([r"(?:gst\s*total\s*amt|total\s*gst|tax\s*amount)\s*[:\-]?\s*([0-9,]+(?:\.\d{1,2})?)"], flattened))
				if gst_total_hint and float(gst_total_hint) > 20:
					half = float(gst_total_hint) / 2.0
					cgst = f"{half:.2f}"
					sgst = f"{half:.2f}"
		except (ValueError, TypeError):
			pass

		# UTGST and SGST are interchangeable (both for state-level tax)
		utgst = clean_amount(find_first([
			r"utgst\s*(?:@|at)?\s*(?:\d{1,2}(?:\.\d+)?\s*%)?\s*([0-9,]+(?:\.\d{1,2})?)"
		], flattened))
		try:
			if utgst and float(utgst) <= 10:
				utgst = ""
		except (ValueError, TypeError):
			pass

		# If SGST/UTGST exist strongly, clear IGST as they are mutually exclusive.
		try:
			if (sgst and float(sgst) > 10) or (utgst and float(utgst) > 10):
				igst = ""
		except (ValueError, TypeError):
			pass

		# Requirement: when SGST/UTGST is present, keep only SGST column populated.
		if utgst:
			if not sgst:
				sgst = utgst
			utgst = ""

		# Amount-led disambiguation: if CGST and SGST are both present and near-equal,
		# prefer state-tax mapping and clear IGST regardless of label noise.
		try:
			cg_num = float(cgst.replace(",", "")) if cgst else 0.0
			sg_num = float(sgst.replace(",", "")) if sgst else 0.0
			if cg_num > 0 and sg_num > 0 and abs(cg_num - sg_num) <= max(2.0, max(cg_num, sg_num) * 0.08):
				igst = ""
		except (ValueError, TypeError):
			pass
	else:
		# IGST-only invoice: clear state-tax fields.
		cgst = ""
		sgst = ""
		utgst = ""

	# Ujjivan GCCP/Micro layouts frequently show only IGST, but OCR may duplicate it
	# into CGST/SGST and miss taxable value. Normalize to IGST-only and backfill taxable.
	try:
		low = flattened.lower()
		if (
			"ujjivan" in low
			and "igst" in low
			and "18" in low
			and not igst
			and cgst
			and sgst
		):
			cg_num = float(cgst.replace(",", ""))
			sg_num = float(sgst.replace(",", ""))
			if cg_num > 10 and sg_num > 10 and abs(cg_num - sg_num) <= max(cg_num, sg_num) * 0.05:
				igst = f"{max(cg_num, sg_num):.2f}"
				cgst = ""
				sgst = ""
				utgst = ""

		if "ujjivan" in low and igst:
			ig_num = float(igst.replace(",", ""))
			total_num = float(amount_total.replace(",", "")) if amount_total else 0.0
			brok_num = float(brokerage.replace(",", "")) if brokerage else 0.0
			if brok_num <= 0 and ig_num > 0:
				# Prefer statutory IGST rate backfill when taxable is missing.
				brok_num = ig_num / 0.18
				brokerage = f"{brok_num:.2f}"
			expected_total = brok_num + ig_num if brok_num > 0 else 0.0
			if expected_total > 0 and (total_num <= 0 or abs(total_num - expected_total) > 2.0):
				amount_total = f"{expected_total:.2f}"
	except (ValueError, TypeError, ZeroDivisionError):
		pass

	# Trusttech invoices are state-tax layouts; avoid mapping total into IGST.
	try:
		if "trusttech" in flattened.lower() and brokerage:
			ct = clean_amount(find_first([
				r"cgst\s*9(?:\.0+)?\s*%\s*[:\-]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
			], flattened))
			st = clean_amount(find_first([
				r"sgst\s*9(?:\.0+)?\s*%\s*[:\-]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
			], flattened))
			if ct and st:
				cgst = ct
				sgst = st
				utgst = ""
				igst = ""
				cg_num = float(cgst.replace(",", ""))
				sg_num = float(sgst.replace(",", ""))
				brok_num = float(brokerage.replace(",", ""))
				gst_num = cg_num + sg_num
				gst_total = f"{gst_num:.2f}"
				expected_total = brok_num + gst_num
				total_num = float(amount_total.replace(",", "")) if amount_total else 0.0
				if total_num <= 0 or abs(total_num - expected_total) > max(2.0, expected_total * 0.02):
					amount_total = f"{expected_total:.2f}"
	except (ValueError, TypeError):
		pass

	# GST TOTAL behavior:
	# - IGST-only invoices => GST TOTAL = IGST
	# - otherwise GST TOTAL = CGST + max(SGST, UTGST)
	gst_total = ""
	try:
		cgst_num = float(cgst.replace(",", "")) if cgst else 0
		sgst_num = float(sgst.replace(",", "")) if sgst else 0
		utgst_num = float(utgst.replace(",", "")) if utgst else 0
		igst_num = float(igst.replace(",", "")) if igst else 0
		state_tax = max(sgst_num, utgst_num)
		if igst_num > 0 and cgst_num == 0 and state_tax == 0:
			gst_total = f"{igst_num:.2f}"
		elif cgst_num + state_tax > 0:
			gst_total = f"{cgst_num + state_tax:.2f}"
		elif cgst_num + igst_num > 0:
			gst_total = f"{cgst_num + igst_num:.2f}"
	except (ValueError, AttributeError):
		pass
	if (not amount_total) and igst_triplet_total:
		amount_total = igst_triplet_total

	has_direct_monetary_basis = bool(re.search(
		r"taxable\s*value|brokerage\s*amount|commission\s*amount|total\s*amount\s*(?:after\s*tax|payable)|gst\s*total\s*amt|igst\s*@|cgst\s*@|sgst\s*@",
		flattened,
		re.IGNORECASE,
	))

	# Deterministic fallback: if taxable and total are present but all GST fields are missing,
	# derive GST from (total - taxable) when the ratio is plausible.
	try:
		brok_num = float(brokerage.replace(",", "")) if brokerage else 0.0
		total_num = float(amount_total.replace(",", "")) if amount_total else 0.0
		cg_num = float(cgst.replace(",", "")) if cgst else 0.0
		sg_num = float(sgst.replace(",", "")) if sgst else 0.0
		ut_num = float(utgst.replace(",", "")) if utgst else 0.0
		ig_num = float(igst.replace(",", "")) if igst else 0.0
		gst_num = float(gst_total.replace(",", "")) if gst_total else 0.0

		if has_direct_monetary_basis and brok_num > 0 and total_num > 0 and (cg_num + sg_num + ut_num + ig_num + gst_num) == 0:
			inferred_gst = total_num - brok_num
			ratio = inferred_gst / brok_num if brok_num else 0.0
			if inferred_gst > 0 and 0.05 <= ratio <= 0.30:
				if has_igst_layout:
					igst = f"{inferred_gst:.2f}"
					cgst = ""
					sgst = ""
					utgst = ""
				else:
					half = inferred_gst / 2.0
					cgst = f"{half:.2f}"
					sgst = f"{half:.2f}"
					utgst = ""
					igst = ""
				gst_total = f"{inferred_gst:.2f}"
	except (ValueError, TypeError, AttributeError, ZeroDivisionError):
		pass

	# Deterministic fallback: if taxable and GST are present but total is missing,
	# derive total as taxable + GST.
	try:
		brok_num = float(brokerage.replace(",", "")) if brokerage else 0.0
		total_num = float(amount_total.replace(",", "")) if amount_total else 0.0
		gst_num = float(gst_total.replace(",", "")) if gst_total else 0.0
		if gst_num <= 0:
			ig_num = float(igst.replace(",", "")) if igst else 0.0
			cg_num = float(cgst.replace(",", "")) if cgst else 0.0
			sg_num = float(sgst.replace(",", "")) if sgst else 0.0
			ut_num = float(utgst.replace(",", "")) if utgst else 0.0
			state_tax = max(sg_num, ut_num)
			gst_num = ig_num if ig_num > 0 else (cg_num + state_tax)
		if has_direct_monetary_basis and brok_num > 0 and gst_num > 0 and total_num <= 0:
			tax_ratio = gst_num / brok_num if brok_num else 0.0
			if 0.05 <= tax_ratio <= 0.30:
				amount_total = f"{(brok_num + gst_num):.2f}"
	except (ValueError, TypeError, AttributeError, ZeroDivisionError):
		pass

	# Deterministic fallback: if brokerage is missing but total and GST are present,
	# derive brokerage as total - GST when ratio is plausible.
	try:
		brok_num = float(brokerage.replace(",", "")) if brokerage else 0.0
		total_num = float(amount_total.replace(",", "")) if amount_total else 0.0
		ig_num = float(igst.replace(",", "")) if igst else 0.0
		cg_num = float(cgst.replace(",", "")) if cgst else 0.0
		sg_num = float(sgst.replace(",", "")) if sgst else 0.0
		ut_num = float(utgst.replace(",", "")) if utgst else 0.0
		state_tax = max(sg_num, ut_num)
		gst_num = float(gst_total.replace(",", "")) if gst_total else 0.0
		if gst_num <= 0:
			gst_num = ig_num if ig_num > 0 else (cg_num + state_tax)

		if has_direct_monetary_basis and brok_num <= 0 and total_num > 0 and gst_num > 0 and total_num > gst_num:
			inferred_brok = total_num - gst_num
			ratio = gst_num / inferred_brok if inferred_brok else 0.0
			if inferred_brok > 0 and 0.05 <= ratio <= 0.30:
				brokerage = f"{inferred_brok:.2f}"
		elif brok_num <= 0 and total_num > 0 and gst_num > 0 and total_num <= (gst_num * 1.05):
			# Total cannot be less than or equal to GST for valid invoices;
			# treat it as mis-mapped tax component and allow tax-only recovery below.
			amount_total = ""
	except (ValueError, TypeError, AttributeError, ZeroDivisionError):
		pass

	# Deterministic fallback: tax-only rows (missing brokerage and total) can be recovered
	# from standard GST rates when one tax mode is clearly present.
	try:
		brok_num = float(brokerage.replace(",", "")) if brokerage else 0.0
		total_num = float(amount_total.replace(",", "")) if amount_total else 0.0
		ig_num = float(igst.replace(",", "")) if igst else 0.0
		cg_num = float(cgst.replace(",", "")) if cgst else 0.0
		sg_num = float(sgst.replace(",", "")) if sgst else 0.0
		ut_num = float(utgst.replace(",", "")) if utgst else 0.0

		if has_direct_monetary_basis and brok_num <= 0 and total_num <= 0:
			# IGST-only recovery.
			if ig_num > 0 and (cg_num + sg_num + ut_num) == 0:
				inferred_brok = ig_num / 0.18
				if inferred_brok > 100:
					brokerage = f"{inferred_brok:.2f}"
					amount_total = f"{(inferred_brok + ig_num):.2f}"
					gst_total = f"{ig_num:.2f}"
			# CGST+SGST/UTGST recovery.
			else:
				state_tax = max(sg_num, ut_num)
				if cg_num > 0 and state_tax > 0 and abs(cg_num - state_tax) <= max(cg_num, state_tax) * 0.2:
					combined = cg_num + state_tax
					inferred_brok = combined / 0.18
					if inferred_brok > 100:
						brokerage = f"{inferred_brok:.2f}"
						amount_total = f"{(inferred_brok + combined):.2f}"
						gst_total = f"{combined:.2f}"
	except (ValueError, TypeError, AttributeError, ZeroDivisionError):
		pass
	# Extract Agent PAN with improved patterns for different layouts
	agent_pan = find_first([
		r"(?:agent\s*)?pan\s*(?:no)?\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])",
		r"(?:p\.a\.n|pan)\s*(?:code)?\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])",
		r"\bPAN\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])",
		r"(?:pan|p\.a\.n)\.?\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])\b",
		r"\b([A-Z]{5}[0-9]{4}[A-Z])\b(?=\s+(?:is|are|the|of|agent))",
	], flattened)
	if not agent_pan:
		broker_pan = gstin_to_pan(broker_gstn)
		if broker_pan and broker_pan != BALIC_PAN:
			agent_pan = broker_pan
	if agent_pan == BALIC_PAN:
		broker_pan = gstin_to_pan(broker_gstn)
		if broker_pan and broker_pan != BALIC_PAN:
			agent_pan = broker_pan
	
	# Arithmetic autofill remains opt-in; suspicious values are validated downstream.

	if allow_arithmetic_autofill():
		# Recalculate GST TOTAL after sanity check
		# Keep extracted tax components unchanged; downstream validation will flag suspicious ratios.
		try:
			brokerage_num = float(brokerage.replace(",", "")) if brokerage else 0
			cgst_num = float(cgst.replace(",", "")) if cgst else 0
			sgst_num = float(sgst.replace(",", "")) if sgst else 0
			_ = (brokerage_num, cgst_num, sgst_num)
		except (ValueError, AttributeError):
			pass

	gst_total = ""
	try:
		cgst_num = float(cgst.replace(",", "")) if cgst else 0
		sgst_num = float(sgst.replace(",", "")) if sgst else 0
		utgst_num = float(utgst.replace(",", "")) if utgst else 0
		igst_num = float(igst.replace(",", "")) if igst else 0
		state_tax = max(sgst_num, utgst_num)
		if igst_num > 0 and cgst_num == 0 and state_tax == 0:
			gst_total = f"{igst_num:.2f}"
		elif cgst_num + state_tax > 0:
			gst_total = f"{cgst_num + state_tax:.2f}"
		elif cgst_num + igst_num > 0:
			gst_total = f"{cgst_num + igst_num:.2f}"
	except (ValueError, AttributeError):
		pass

	# Mapping normalization:
	# 1) If IGST was duplicated into CGST/SGST, keep only IGST.
	# 2) If Total Inv Amt is actually a tax component, replace with brokerage + GST.
	try:
		brok_num = float(brokerage.replace(",", "")) if brokerage else 0
		total_num = float(amount_total.replace(",", "")) if amount_total else 0
		cgst_num = float(cgst.replace(",", "")) if cgst else 0
		sgst_num = float(sgst.replace(",", "")) if sgst else 0
		utgst_num = float(utgst.replace(",", "")) if utgst else 0
		igst_num = float(igst.replace(",", "")) if igst else 0

		# Collapse duplicate IGST represented across all tax columns.
		if igst_num > 0 and cgst_num > 0 and sgst_num > 0:
			same_state = abs(cgst_num - sgst_num) <= max(cgst_num, sgst_num) * 0.05
			same_as_igst = abs(cgst_num - igst_num) <= max(cgst_num, igst_num) * 0.05
			if same_state and same_as_igst:
				cgst = ""
				sgst = ""
				utgst = ""
				gst_total = f"{igst_num:.2f}"
				if total_num > igst_num and brok_num > total_num:
					brok_num = total_num - igst_num
					brokerage = f"{brok_num:.2f}"

		# Some bank layouts leak IGST into only one state-tax column (CGST or SGST),
		# which can incorrectly double GST TOTAL when both are summed later.
		if igst_num > 0 and cgst_num > 0 and sgst_num == 0 and abs(cgst_num - igst_num) <= max(cgst_num, igst_num) * 0.05:
			cgst = ""
			utgst = ""
			gst_total = f"{igst_num:.2f}"
		if igst_num > 0 and sgst_num > 0 and cgst_num == 0 and abs(sgst_num - igst_num) <= max(sgst_num, igst_num) * 0.05:
			sgst = ""
			utgst = ""
			gst_total = f"{igst_num:.2f}"

		# Correct totals that are actually one tax component (common mapping issue in Bajaj docs).
		gst_num = float(gst_total.replace(",", "")) if gst_total else 0
		tax_component = max(cgst_num, sgst_num, utgst_num, igst_num)
		if brok_num > 0 and gst_num > 0 and total_num > 0:
			if total_num <= brok_num * 1.02 and (abs(total_num - tax_component) <= max(1.0, tax_component * 0.05) or total_num <= tax_component * 1.05):
				amount_total = f"{(brok_num + gst_num):.2f}"
	except (ValueError, AttributeError, ZeroDivisionError):
		pass

	if allow_arithmetic_autofill():
		# Final monetary backfill when OCR misses labels but tax components are present.
		try:
			cgst_num = float(cgst.replace(",", "")) if cgst else 0
			sgst_num = float(sgst.replace(",", "")) if sgst else 0
			utgst_num = float(utgst.replace(",", "")) if utgst else 0
			igst_num = float(igst.replace(",", "")) if igst else 0
			brokerage_num = float(brokerage.replace(",", "")) if brokerage else 0
			gst_total_num = float(gst_total.replace(",", "")) if gst_total else 0

			state_component = max(sgst_num, utgst_num)
			if brokerage_num == 0 and cgst_num > 10 and state_component > 10:
				if abs(cgst_num - state_component) <= max(cgst_num, state_component) * 0.25:
					brokerage_num = cgst_num / 0.09
					brokerage = f"{brokerage_num:.2f}"

			total_num = float(amount_total.replace(",", "")) if amount_total else 0
			if brokerage_num == 0 and total_num > 0:
				if igst_num > 0 and total_num > igst_num:
					brokerage_num = total_num - igst_num
					brokerage = f"{brokerage_num:.2f}"
				elif gst_total_num > 0 and total_num > gst_total_num:
					brokerage_num = total_num - gst_total_num
					brokerage = f"{brokerage_num:.2f}"
				elif (cgst_num + state_component) > 0 and total_num > (cgst_num + state_component):
					brokerage_num = total_num - (cgst_num + state_component)
					brokerage = f"{brokerage_num:.2f}"

			if not amount_total and brokerage_num > 0:
				if gst_total_num > 0:
					amount_total = f"{(brokerage_num + gst_total_num):.2f}"
				elif igst_num > 0:
					amount_total = f"{(brokerage_num + igst_num):.2f}"
				elif (cgst_num + state_component) > 0:
					amount_total = f"{(brokerage_num + cgst_num + state_component):.2f}"
		except (ValueError, AttributeError, ZeroDivisionError):
			pass

	if allow_arithmetic_autofill():
		# Backfill from tax amount in words when numeric tax labels are noisy.
		try:
			tax_words_num = float(tax_total_from_words.replace(",", "")) if tax_total_from_words else 0
			if tax_words_num > 0:
				if (not gst_total) or (float(gst_total.replace(",", "")) <= 0):
					gst_total = f"{tax_words_num:.2f}"
				if not has_igst_layout and not igst:
					if (not cgst) or (not sgst) or abs(float(cgst or 0) - float(sgst or 0)) > tax_words_num * 0.6:
						half = tax_words_num / 2.0
						cgst = f"{half:.2f}"
						sgst = f"{half:.2f}"
				if amount_total:
					amount_num = float(amount_total.replace(",", ""))
					if amount_num > tax_words_num and not brokerage:
						brokerage = f"{(amount_num - tax_words_num):.2f}"
				elif brokerage and not amount_total:
					brokerage_num = float(brokerage.replace(",", ""))
					amount_total = f"{(brokerage_num + tax_words_num):.2f}"
		except (ValueError, AttributeError, ZeroDivisionError):
			pass

	# Finozone-specific consistency: use amount-in-words and tax-in-words as source of truth.
	if allow_arithmetic_autofill() and "finozone" in flattened.lower():
		try:
			total_num = float(amount_total.replace(",", "")) if amount_total else 0
			tax_words_num = float(tax_total_from_words.replace(",", "")) if tax_total_from_words else 0
			if total_num > 0 and tax_words_num > 0:
				# Set GST split from tax words to avoid address-number contamination.
				half = tax_words_num / 2.0
				cgst = f"{half:.2f}"
				sgst = f"{half:.2f}"
				utgst = ""
				igst = ""
				gst_total = f"{tax_words_num:.2f}"
				if total_num > tax_words_num:
					brokerage = f"{(total_num - tax_words_num):.2f}"
		except (ValueError, AttributeError, ZeroDivisionError):
			pass

	# Medwell mapping correction: these invoices are IGST-only.
	# If CGST/SGST were both populated from the same IGST value, collapse to IGST.
	if "medwell insurance broking" in flattened.lower():
		try:
			# Prefer direct row mapping from OCR text: <taxable> 18% <igst> <total>
			med_match = None
			for m in re.finditer(
				r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+18\s*%?\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s+([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
				flattened,
				re.IGNORECASE,
			):
				taxable_v = float(clean_amount(m.group(1)) or "0")
				igst_v = float(clean_amount(m.group(2)) or "0")
				total_v = float(clean_amount(m.group(3)) or "0")
				if taxable_v > 0 and igst_v > 0 and total_v >= taxable_v and total_v > igst_v:
					med_match = m
					break
			if med_match:
				med_taxable = clean_amount(med_match.group(1))
				med_igst = clean_amount(med_match.group(2))
				med_total = clean_amount(med_match.group(3))
				if med_taxable:
					brokerage = med_taxable
				if med_igst:
					igst = med_igst
					gst_total = med_igst
				if med_total:
					amount_total = med_total

			cg_num = float(cgst.replace(",", "")) if cgst else 0
			sg_num = float(sgst.replace(",", "")) if sgst else 0
			ig_num = float(igst.replace(",", "")) if igst else 0
			gst_num = float(gst_total.replace(",", "")) if gst_total else 0

			if ig_num <= 0 and cg_num > 0 and sg_num > 0 and abs(cg_num - sg_num) <= max(cg_num, sg_num) * 0.05:
				ig_num = max(cg_num, sg_num)
				igst = f"{ig_num:.2f}"

			if ig_num <= 0 and gst_num > 0 and cg_num == 0 and sg_num == 0:
				ig_num = gst_num
				igst = f"{ig_num:.2f}"

			# If total-brokerage is about 2x IGST, invoice behaves like CGST+SGST layout.
			brok_num = float(brokerage.replace(",", "")) if brokerage else 0.0
			total_num = float(amount_total.replace(",", "")) if amount_total else 0.0
			if ig_num > 0 and brok_num > 0 and total_num > brok_num:
				implied_tax = total_num - brok_num
				if abs(implied_tax - (2.0 * ig_num)) <= max(2.0, implied_tax * 0.05):
					cgst = f"{ig_num:.2f}"
					sgst = f"{ig_num:.2f}"
					utgst = ""
					igst = ""
					gst_total = f"{implied_tax:.2f}"
				else:
					cgst = ""
					sgst = ""
					utgst = ""
					gst_total = f"{ig_num:.2f}"
			elif ig_num > 0:
				cgst = ""
				sgst = ""
				utgst = ""
				gst_total = f"{ig_num:.2f}"
		except (ValueError, AttributeError, ZeroDivisionError):
			pass

	# Bajaj mapping correction: some layouts map taxable value into Total Inv Amt.
	# When Total Inv Amt looks like a tax component, clear it (do not compute).
	if extract_bajaj_company_name(flattened):
		try:
			total_num = float(amount_total.replace(",", "")) if amount_total else 0
			cg_num = float(cgst.replace(",", "")) if cgst else 0
			sg_num = float(sgst.replace(",", "")) if sgst else 0
			ut_num = float(utgst.replace(",", "")) if utgst else 0
			ig_num = float(igst.replace(",", "")) if igst else 0
			tax_component = max(cg_num, sg_num, ut_num, ig_num)
			if total_num > 0 and (abs(total_num - tax_component) <= max(1.0, tax_component * 0.05) or total_num <= tax_component * 1.05):
				amount_total = ""
		except (ValueError, AttributeError, ZeroDivisionError):
			pass

	if allow_arithmetic_autofill():
		# Final normalization: prevent CGST/SGST or IGST rate values (9/18) from surviving as amounts.
		try:
			brokerage_num = float(brokerage.replace(",", "")) if brokerage else 0
			cgst_num = float(cgst.replace(",", "")) if cgst else 0
			sgst_num = float(sgst.replace(",", "")) if sgst else 0
			igst_num = float(igst.replace(",", "")) if igst else 0
			if brokerage_num > 100:
				if cgst_num > 0 and sgst_num > 0 and cgst_num <= 30 and sgst_num <= 30:
					expected = brokerage_num * 0.09
					cgst = f"{expected:.2f}"
					sgst = f"{expected:.2f}"
					utgst = ""
					igst = ""
					gst_total = f"{(expected * 2):.2f}"
				elif igst_num > 0 and igst_num <= 30 and cgst_num == 0 and sgst_num == 0:
					expected_igst = brokerage_num * 0.18
					igst = f"{expected_igst:.2f}"
					gst_total = f"{expected_igst:.2f}"
		except (ValueError, AttributeError):
			pass

	# Source-specific normalization for layouts that frequently mis-map total from noisy tokens.
	try:
		brok_num = float(brokerage.replace(",", "")) if brokerage else 0.0
		igst_num = float(igst.replace(",", "")) if igst else 0.0
		cgst_num = float(cgst.replace(",", "")) if cgst else 0.0
		sgst_num = float(sgst.replace(",", "")) if sgst else 0.0
		utgst_num = float(utgst.replace(",", "")) if utgst else 0.0
		total_num = float(amount_total.replace(",", "")) if amount_total else 0.0
		if brok_num > 0 and igst_num > 0:
			expected_total = brok_num + igst_num
			low = flattened.lower()
			if "ethika insurance" in low:
				if total_num <= brok_num or abs(total_num - expected_total) > max(2.0, expected_total * 0.05):
					amount_total = f"{expected_total:.2f}"
			if "city union bank" in low:
				if total_num <= brok_num or abs(total_num - expected_total) > max(2.0, expected_total * 0.08):
					amount_total = f"{expected_total:.2f}"

		# Axis Bank invoices can mis-map total from nearby HSN/row constants (e.g., 997119).
		# Prefer deterministic total from taxable + GST when parsed total is inconsistent.
		low = flattened.lower()
		if ("axis bank" in low or ("axis" in low and "taxable value of supply" in low)) and brok_num > 0:
			state_tax = max(sgst_num, utgst_num)
			gst_component = igst_num if igst_num > 0 else (cgst_num + state_tax)
			if gst_component > 0:
				expected_total = brok_num + gst_component
				if abs(total_num - 997119.0) <= 2.0 or abs(total_num - 997161.0) <= 2.0:
					amount_total = f"{expected_total:.2f}"
					total_num = expected_total
				if total_num <= brok_num or abs(total_num - expected_total) > max(2.0, expected_total * 0.05):
					amount_total = f"{expected_total:.2f}"

		# Generic high-mismatch guard: when taxable and GST are both parsed,
		# a total that deviates heavily is usually an OCR/HSN mapping artifact.
		if brok_num > 0:
			state_tax = max(sgst_num, utgst_num)
			gst_component = igst_num if igst_num > 0 else (cgst_num + state_tax)
			if gst_component > 0 and total_num > 0:
				expected_total = brok_num + gst_component
				if abs(total_num - expected_total) > max(10.0, expected_total * 0.20):
					amount_total = f"{expected_total:.2f}"
	except (ValueError, AttributeError, ZeroDivisionError):
		pass

	if allow_arithmetic_autofill():
		# Motilal local rows can lose one decimal place in OCR (e.g., 653954.41 -> 6539544.05).
		try:
			if "motilal oswal financial services" in flattened.lower() and amount_total and brokerage and gst_total:
				total_num = float(amount_total.replace(",", ""))
				brokerage_num = float(brokerage.replace(",", ""))
				gst_total_num = float(gst_total.replace(",", ""))
				expected_total = brokerage_num + gst_total_num
				if expected_total > 0:
					ratio = total_num / expected_total
					if 9.5 <= ratio <= 10.5:
						amount_total = f"{expected_total:.2f}"
		except (ValueError, AttributeError, ZeroDivisionError):
			pass

	# India Post Payments Bank uses a distinct tax table; map it explicitly.
	if "india post payments bank" in flattened.lower():
		ip_taxable, ip_cgst, ip_sgst, ip_igst, ip_total = extract_india_post_amounts(flattened)
		if ip_taxable:
			brokerage = ip_taxable
		if ip_total:
			amount_total = ip_total
		if ip_igst and float(ip_igst.replace(",", "")) > 0:
			igst = ip_igst
			cgst = ""
			sgst = ""
			utgst = ""
			gst_total = ip_igst
		else:
			if ip_cgst:
				cgst = ip_cgst
			if ip_sgst:
				sgst = ip_sgst
			utgst = ""
			igst = ""
			try:
				gst_total = f"{(float(cgst.replace(',', '')) if cgst else 0) + (float(sgst.replace(',', '')) if sgst else 0):.2f}"
			except (ValueError, TypeError):
				gst_total = ""

		customer_name = sanitize_party_name(find_first([
			r"customer\s*name\s*[:\-]?\s*([A-Za-z][A-Za-z\s&.,()'/-]{4,80}?(?:Limited|Ltd))",
		], flattened))
		if customer_name:
			service_recipient = normalize_balic_service_recipient_name(customer_name, flattened)

		ip_date = find_first([
			r"invoice\s*date\s*[:\-]?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
		], flattened)
		if ip_date:
			vendor_date = try_parse_date(ip_date)

		if not invoice_no:
			ip_ref = find_first([
				r"invoice\s*reference\s*no\s*[:\-]?\s*([A-Z0-9\-/]{6,30})",
			], flattened)
			if ip_ref and is_valid_invoice_no(ip_ref):
				invoice_no = ip_ref

	# Livlong/IIFL layout: explicit commission + IGST line can be overshadowed by noisy table values.
	if (
		"livlong insurance brokers limited" in flattened.lower()
		or "iifl insurance brokers limited" in flattened.lower()
	):
		liv_brokerage = clean_amount(find_first([
			r"commission\s+for\s+the\s+month.*?([0-9][0-9,]*(?:\.[0-9]{1,2}))",
		], flattened))
		liv_igst = clean_amount(find_first([
			r"add\s*[:\-]*\s*igst\s*@?\s*18\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		], flattened))
		liv_cgst = clean_amount(find_first([
			r"add\s*[:\-]*\s*cgst\s*@?\s*9\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		], flattened))
		liv_sgst = clean_amount(find_first([
			r"add\s*[:\-]*\s*sgst\s*@?\s*9\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		], flattened))
		liv_total = clean_amount(find_first([
			r"\btotal\b[^0-9]{0,12}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
		], flattened))
		try:
			liv_b_num = float(liv_brokerage) if liv_brokerage else 0.0
			liv_i_num = float(liv_igst) if liv_igst else 0.0
			liv_c_num = float(liv_cgst) if liv_cgst else 0.0
			liv_s_num = float(liv_sgst) if liv_sgst else 0.0
			liv_gst_num = liv_i_num if liv_i_num > 0 else (liv_c_num + liv_s_num)
			if liv_b_num > 0 and (liv_i_num > 0 or (liv_c_num > 0 and liv_s_num > 0)):
				brokerage = liv_brokerage
				if liv_i_num > 0:
					igst = liv_igst
					cgst = ""
					sgst = ""
					utgst = ""
					gst_total = liv_igst
				else:
					igst = ""
					cgst = liv_cgst
					sgst = liv_sgst
					utgst = ""
					gst_total = f"{(liv_c_num + liv_s_num):.2f}"
				if liv_total and float(liv_total) > liv_gst_num:
					amount_total = liv_total
		except (ValueError, TypeError):
			pass

	# Final guard: if all tax columns carry the same value, treat it as IGST-only mapping.
	try:
		cg_num = float(cgst.replace(",", "")) if cgst else 0
		sg_num = float(sgst.replace(",", "")) if sgst else 0
		ig_num = float(igst.replace(",", "")) if igst else 0
		if cg_num > 0 and sg_num > 0 and ig_num > 0:
			same_state = abs(cg_num - sg_num) <= max(cg_num, sg_num) * 0.05
			same_all = abs(cg_num - ig_num) <= max(cg_num, ig_num) * 0.05
			if same_state and same_all:
				cgst = ""
				sgst = ""
				utgst = ""
				gst_total = f"{ig_num:.2f}"
	except (ValueError, AttributeError, ZeroDivisionError):
		pass

	# Guard against HSN/SAC or taxable values leaking into CGST/SGST columns.
	try:
		brok_num = float(brokerage.replace(",", "")) if brokerage else 0.0
		cg_num = float(cgst.replace(",", "")) if cgst else 0.0
		sg_num = float(sgst.replace(",", "")) if sgst else 0.0
		if cg_num >= 900000 and cg_num <= 999999:
			cgst = ""
		if sg_num >= 900000 and sg_num <= 999999:
			sgst = ""
		if brok_num > 0:
			if cg_num >= brok_num * 0.85:
				cgst = ""
			if sg_num >= brok_num * 0.85:
				sgst = ""
	except (ValueError, TypeError, AttributeError):
		pass

	# Guard against HSN/SAC leakage in brokerage amount.
	try:
		brok_num = float(brokerage.replace(",", "")) if brokerage else 0.0
		if 900000 <= brok_num <= 999999:
			brokerage = ""
		if brokerage in {"997161", "997119", "997116"}:
			brokerage = ""
	except (ValueError, TypeError, AttributeError):
		pass
	if not agent_pan:
		agent_pan = find_first([r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"], flattened)

	# Extract SAC code with label anchoring to avoid table-row HSN leakage.
	sac_code = find_first([
		r"(?:service\s+accounting\s+code|sac\s*code)\s*[:\-]?\s*([0-9]{4,8})",
		r"\b(?:sac)\s*[:\-]?\s*([0-9]{6,8})\b",
	], flattened)
	
	# Extract date ranges from narration/header.
	date_from = ""
	date_to = ""
	date_range_match = re.search(r"(\d{2})\s+(?:to|through)\s+(\d{2})\s+([A-Za-z]{3,9})['\"]?([0-9]{2,4})", flattened)
	if date_range_match:
		day_from, day_to, month_name, year = date_range_match.groups()
		if len(year) == 2:
			year = f"20{year}"
		month_map = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
		            "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
		month_num = month_map.get(month_name[:3].lower(), "")
		if month_num:
			date_from = f"{day_from}/{month_num}/{year}"
			date_to = f"{day_to}/{month_num}/{year}"

	if not date_from or not date_to:
		dmy_range = re.search(
			r"([0-3]?\d[./-][01]?\d[./-](?:\d{2}|\d{4}))\s*(?:to|through|till|-)\s*([0-3]?\d[./-][01]?\d[./-](?:\d{2}|\d{4}))",
			flattened,
			re.IGNORECASE,
		)
		if dmy_range:
			date_from = try_parse_date(dmy_range.group(1))
			date_to = try_parse_date(dmy_range.group(2))

	if not date_from or not date_to:
		month_period = re.search(
			r"\b(?:for\s+the\s+month\s+of|month\s+of|period\s+of|commission\s+for\s+the\s+month\s+of|brokerage\s+for\s+the\s+month\s+of)\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\s*['’\-/]?\s*(\d{2,4})\b",
			flattened,
			re.IGNORECASE,
		)
		if month_period:
			mon, year = month_period.groups()
			month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
			month_num = month_map.get(mon[:3].lower(), 0)
			if month_num:
				year_num = int(year)
				if year_num < 100:
					year_num += 2000
				last_day = calendar.monthrange(year_num, month_num)[1]
				date_from = f"01/{month_num:02d}/{year_num}"
				date_to = f"{last_day:02d}/{month_num:02d}/{year_num}"
	
	# Fallback to generic date range extraction
	if not date_from:
		date_from = try_parse_date(find_first([r"(?:date\s*from|period\s*from|from)\s*[:\-]?\s*([0-9./\-]{6,12})"], flattened))
	if not date_to:
		date_to = try_parse_date(find_first([r"(?:date\s*to|period\s*to|to)\s*[:\-]?\s*([0-9./\-]{6,12})"], flattened))
	
	# Extract Narration from meaningful commission/brokerage phrases only.
	narration = ""
	narration_patterns = [
		r"\b(BALIC\s+COMM[^|]{0,120})",
		r"\b(Brokerage\s+Commi(?:ss|s)sion\s+for\s+the\s+Month\s+of[^|]{0,140})",
		r"\b(Commission\s+for\s+the\s+Month\s+of[^|]{0,140})",
		r"\b(Brokerage\s+for\s+M/?o\s+[A-Za-z]{3,9}\s*[0-9]{2,4}[^|]{0,80})",
	]
	narration_blacklist = re.compile(
		r"\b(total\s*amount|igst|sgst|utgst|cgst|taxable\s*value|rate\b|hsn|sac|quantity|unit\s*price|other\s+financial\s+services)\b",
		re.IGNORECASE,
	)
	for pattern in narration_patterns:
		for m in re.finditer(pattern, flattened, re.IGNORECASE):
			can = re.sub(r"\s+", " ", (m.group(1) or "")).strip(" :-")
			if len(can) < 8:
				continue
			if narration_blacklist.search(can):
				continue
			narration = can
			break
		if narration:
			break

	service_recipient = find_first(
		[
			r"(?:branch\s*(?:name|address)|Name.*?Address)\s*:?\s*([A-Z][a-zA-Z\s&.,()-]{8,60}?)(?:\s+(?:Contact|CIN|Address|GSTIN|State|Phone|Near))",
			r"(?:Service\s*Recipient|BAJAJ\s+LIFE|Service\s+Recipient\s+Name)\s*:?\s*([A-Z][A-Za-z\s&.,()-]{8,60}?)(?:\s+(?:Address|Near|GSTIN))",
			r"(?:name\s*of\s*service\s*rec(?:e|i)pient)\s*[:\-]?\s*([A-Z0-9 .,&()-]{3,80})",
		],
		flattened,
	)
	if not service_recipient:
		service_recipient = extract_company_from_address_start(flattened)

	agent_name = find_first([
		r"agent\s*name\s*[:\-]?\s*([A-Z][A-Za-z .,&()-]{3,})",
	], flattened)
	if not agent_name:
		agent_name = extract_bajaj_company_name(flattened)
	if not agent_name:
		agent_name = service_recipient

	agent_name = normalize_balic_company_name(agent_name)
	service_recipient = normalize_balic_service_recipient_name(service_recipient, flattened)

	balic_state = extract_state_from_text(flattened)
	broker_state_raw = find_first([
		r"broker\s*gstn\s*state\s*[:\-]?\s*([A-Z ]{2,})",
		r"broker\s*state\s*[:\-]?\s*([A-Za-z ]{2,40})",
	], flattened)
	broker_state = normalize_state_name(broker_state_raw)
	if balic_gstn:
		balic_state = gstin_to_state_name(balic_gstn)
	if broker_gstn:
		broker_state = gstin_to_state_name(broker_gstn)
	if not balic_state and broker_state:
		balic_state = broker_state

	# Try to extract AGENT_CODE from text, or use similarity matching
	extracted_agent_code = find_first([
		r"(?:agent\s*code|ag\.?\s*code|agent\s*id|code)\s*[:\-]?\s*([A-Z0-9\-/]+)",
	], flattened)
	
	# If not found in text, try similarity matching on agent name
	final_agent_code = extracted_agent_code
	final_agent_name = agent_name
	
	if not final_agent_code and agent_name:
		matched_code, clean_name, similarity = find_best_matching_agent_code(agent_name)
		if matched_code:
			final_agent_code = matched_code
			final_agent_name = clean_name
			LOGGER.debug("Matched agent '%s' to code %s (similarity: %.2f)", agent_name, matched_code, similarity)

	row = {
		"AGENT_CODE": final_agent_code,
		"Agent Name": final_agent_name,
		"Agent PAN": agent_pan,
		"Name of Service Receipient": normalize_balic_service_recipient_name(service_recipient, flattened),
		"BALIC STATE": balic_state,
		"BALIC GSTN": balic_gstn,
		"BROKER GSTN STATE": broker_state,
		"BROKER GSTN": broker_gstn,
		"Vendor Inv Date": vendor_date,
		"Vendor Inv No": invoice_no,
		"Total Inv Amt": amount_total,
		"BROKERAGE Amount": brokerage,
		"CGST @ 9%": cgst,
		"SGST @ 9%": sgst,
		"UTGST": utgst,
		"IGST": igst,
		"GST TOTAL AMT": gst_total,
		"DATE_FROM": date_from,
		"DATE_TO": date_to,
		"Narration": narration if narration else "COMMISSION",
		"Type": find_first([
			r"(?:commission|type)\s*[:\-]?\s*([A-Z][A-Z\s]{2,15}?)(?:\s*[_\-]|\s+Non|\s+Micro|$)",
			r"([A-Z][A-Z]{2,15})\s*(?:INDIVIDUAL|GROUP|CORPORATE)",
			r"\b(Individual|Group|Corporate)\b"],
			flattened),
		"Micro/Non Micro": "",
		"SAC Code": sac_code,
	}
	if row["AGENT_CODE"] and not re.search(r"\d", row["AGENT_CODE"]):
		row["AGENT_CODE"] = ""
	if row["Agent PAN"] == BALIC_PAN:
		row["Agent PAN"] = ""

	if not row["Type"]:
		row["Type"] = find_first([r"\btype\s*[:\-]?\s*([A-Z ]{3,})"], flattened)

	for col in OUTPUT_COLUMNS:
		row.setdefault(col, "")

	return row


def has_meaningful_data(row: Dict[str, str]) -> bool:
	invoice_ok = is_valid_invoice_no(row.get("Vendor Inv No", ""))
	date_ok = bool((row.get("Vendor Inv Date", "") or "").strip())
	amount_ok = False
	for field in ["Total Inv Amt", "BROKERAGE Amount", "GST TOTAL AMT", "CGST @ 9%", "SGST @ 9%", "IGST"]:
		val = clean_amount(row.get(field, ""))
		if not val:
			continue
		try:
			if float(val.replace(",", "")) > 0:
				amount_ok = True
				break
		except (ValueError, AttributeError):
			continue
	pan_ok = bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", (row.get("Agent PAN", "") or "").strip(), re.IGNORECASE))
	balic_gstn_ok = bool(re.fullmatch(r"[0-9A-Z]{15}", (row.get("BALIC GSTN", "") or "").strip(), re.IGNORECASE))
	broker_gstn_ok = bool(re.fullmatch(r"[0-9A-Z]{15}", (row.get("BROKER GSTN", "") or "").strip(), re.IGNORECASE))
	sac_ok = bool(re.fullmatch(r"[0-9]{4,8}", (row.get("SAC Code", "") or "").strip()))

	# Business rule from user: invoice number should always be present.
	if not invoice_ok:
		return False

	if invoice_ok and (date_ok or amount_ok or pan_ok or balic_gstn_ok or broker_gstn_ok or sac_ok):
		return True

	signals = [date_ok, amount_ok, pan_ok, balic_gstn_ok, broker_gstn_ok, sac_ok]
	return any(signals)


def is_actual_receipt_row(fields: Dict[str, str], receipt_text: str) -> bool:
	"""Return True only when chunk looks like an actual receipt line item."""
	invoice_ok = is_valid_invoice_no(fields.get("Vendor Inv No", ""))
	date_ok = bool((fields.get("Vendor Inv Date", "") or "").strip())
	has_amount = any(
		bool(clean_amount(fields.get(k, "")))
		for k in ["Total Inv Amt", "BROKERAGE Amount", "GST TOTAL AMT", "CGST @ 9%", "SGST @ 9%", "IGST"]
	)
	meaningful_identity = any(
		bool((fields.get(k, "") or "").strip())
		for k in ["Agent Name", "AGENT_CODE", "Agent PAN", "Name of Service Receipient", "BALIC GSTN", "BROKER GSTN"]
	)
	receipt_label = bool(
		re.search(
			r"\b(invoice\s*(?:no\.?|number|reference\s*no)|bill\s*(?:no\.?|number)|document\s*no\.?)\b",
			receipt_text or "",
			re.IGNORECASE,
		)
	)
	if not (invoice_ok or date_ok or has_amount or meaningful_identity):
		return False
	if invoice_ok:
		return date_ok or has_amount or receipt_label
	# Allow monetary-only continuation rows; invoice can be recovered later by cross-page reconciliation.
	return has_amount and (date_ok or receipt_label)


def _is_bajaj_finance_context(text: str) -> bool:
	raw = (text or "").lower()
	has_finance = bool(re.search(r"\bbajaj\s+finance\b", raw))
	has_housing = bool(re.search(r"\bbajaj\s+housing\b", raw))
	return has_finance and not has_housing


def _is_bajaj_finance_row(fields: Dict[str, str]) -> bool:
	candidates = " ".join(
		[
			(fields.get("Name of Service Receipient", "") or ""),
			(fields.get("Agent Name", "") or ""),
			(fields.get("Narration", "") or ""),
		]
	)
	if _is_bajaj_finance_context(candidates):
		return True
	company = extract_bajaj_company_name(candidates)
	return normalize_bajaj_company_name(company) == "Bajaj Finance Limited"


def _is_bajaj_finance_total_row(receipt_text: str, fields: Dict[str, str]) -> bool:
	haystack = " ".join(
		[
			receipt_text or "",
			(fields.get("Agent Name", "") or ""),
			(fields.get("Narration", "") or ""),
			(fields.get("Name of Service Receipient", "") or ""),
		]
	)
	if not _is_bajaj_finance_context(haystack) and not _is_bajaj_finance_row(fields):
		return False
	particulars_total = bool(
		re.search(r"\bparticulars?\b.{0,120}\btotal\b|\btotal\b.{0,120}\bparticulars?\b", haystack, re.IGNORECASE | re.DOTALL)
	)
	row_has_amount = any(
		bool(clean_amount(fields.get(k, "")))
		for k in ["Total Inv Amt", "BROKERAGE Amount", "GST TOTAL AMT", "CGST @ 9%", "SGST @ 9%", "IGST"]
	)
	return particulars_total and row_has_amount


def _amount_to_float(value: str) -> float:
	text = (clean_amount(value) or "").replace(",", "").strip()
	if not text:
		return 0.0
	try:
		return float(text)
	except (TypeError, ValueError):
		return 0.0


def _monetary_profile(row: Dict[str, str]) -> Tuple[float, float, float]:
	total = _amount_to_float(row.get("Total Inv Amt", ""))
	brok = _amount_to_float(row.get("BROKERAGE Amount", ""))
	cgst = _amount_to_float(row.get("CGST @ 9%", ""))
	sgst = _amount_to_float(row.get("SGST @ 9%", ""))
	utgst = _amount_to_float(row.get("UTGST", ""))
	igst = _amount_to_float(row.get("IGST", ""))
	gst_total = _amount_to_float(row.get("GST TOTAL AMT", ""))
	state_tax = max(sgst, utgst)
	tax = igst if igst > 0 else (cgst + state_tax)
	if tax <= 0 and gst_total > 0:
		tax = gst_total
	return total, brok, tax


def _same_line_item_by_amount(current_row: Dict[str, str], previous_row: Dict[str, str]) -> bool:
	"""Return True when two rows look like the same line item by monetary signature."""
	if not previous_row:
		return False
	cur_total, cur_brok, cur_tax = _monetary_profile(current_row)
	prev_total, prev_brok, prev_tax = _monetary_profile(previous_row)

	if cur_total > 0 and prev_total > 0 and abs(cur_total - prev_total) <= 1.0:
		return True
	if cur_brok > 0 and prev_brok > 0 and abs(cur_brok - prev_brok) <= 1.0:
		if cur_tax > 0 and prev_tax > 0:
			return abs(cur_tax - prev_tax) <= 1.0
		return True
	return False


def _is_high_value_row(row: Dict[str, str]) -> bool:
	total = _amount_to_float(row.get("Total Inv Amt", ""))
	brok = _amount_to_float(row.get("BROKERAGE Amount", ""))
	gst = _amount_to_float(row.get("GST TOTAL AMT", ""))
	return total >= 200000 or brok >= 150000 or gst >= 30000


def _should_ai_override_monetary(existing: Dict[str, str], ai_row: Dict[str, str]) -> bool:
	"""Use LLM row as validator/corrector only when regex row fails and AI row passes."""
	try:
		existing_valid, _ = validate_math_extraction(existing)
		ai_valid, _ = validate_math_extraction(ai_row)
		if (not existing_valid) and ai_valid:
			return True
	except Exception:
		return False
	return False


def _should_ai_override_identity(existing: Dict[str, str], ai_row: Dict[str, str]) -> bool:
	"""Allow identity replacement only when AI resolves a detected GSTIN-role conflict."""
	return _has_gstin_role_conflict(existing) and (not _has_gstin_role_conflict(ai_row))


def _monetary_signal_score(row: Dict[str, str]) -> int:
	score = 0
	if _amount_to_float(row.get("BROKERAGE Amount", "")) > 0:
		score += 2
	if _amount_to_float(row.get("Total Inv Amt", "")) > 0:
		score += 2
	if _amount_to_float(row.get("GST TOTAL AMT", "")) > 0:
		score += 2
	if _detect_clear_tax_mode(row) in {"igst", "state"}:
		score += 2
	try:
		ok, _ = validate_math_extraction(row)
		if ok:
			score += 4
	except Exception:
		pass
	return score


def _backfill_brokerage_from_total_and_gst(row: Dict[str, str]) -> Dict[str, str]:
	result = dict(row)
	brok = _amount_to_float(result.get("BROKERAGE Amount", ""))
	total = _amount_to_float(result.get("Total Inv Amt", ""))
	gst_total = _amount_to_float(result.get("GST TOTAL AMT", ""))
	if brok <= 0 and total > 0 and gst_total > 0 and total > gst_total:
		inferred = total - gst_total
		if inferred > 0:
			ratio = gst_total / inferred
			if 0.16 <= ratio <= 0.20:
				result["BROKERAGE Amount"] = _fmt_amount(inferred)
	return result


def _reconcile_regex_ai_monetary(existing: Dict[str, str], ai_row: Dict[str, str]) -> Dict[str, str]:
	"""Cross-verify regex and AI rows; keep the stronger monetary mapping and backfill gaps from the other."""
	mcols = [
		"BROKERAGE Amount",
		"Total Inv Amt",
		"GST TOTAL AMT",
		"CGST @ 9%",
		"SGST @ 9%",
		"UTGST",
		"IGST",
	]

	existing_norm = apply_confident_math_fill(enforce_tax_mode_fields(_backfill_brokerage_from_total_and_gst(existing)))
	ai_norm = apply_confident_math_fill(enforce_tax_mode_fields(_backfill_brokerage_from_total_and_gst(ai_row)))

	existing_score = _monetary_signal_score(existing_norm)
	ai_score = _monetary_signal_score(ai_norm)

	primary = existing_norm if existing_score >= ai_score else ai_norm
	secondary = ai_norm if primary is existing_norm else existing_norm

	result = dict(primary)
	for col in mcols:
		if not (result.get(col, "") or "").strip() and (secondary.get(col, "") or "").strip():
			result[col] = secondary[col]

	result = apply_confident_math_fill(result)
	result = enforce_tax_mode_fields(result)
	result = apply_gst_autofill(result)
	return result


def merge_page_fallback_fields(row: Dict[str, str], page_fields: Dict[str, str]) -> Dict[str, str]:
	result = dict(row)
	fill_columns = [
		"Vendor Inv No",
		"Vendor Inv Date",
		"Agent Name",
		"Name of Service Receipient",
		"BALIC STATE",
		"BALIC GSTN",
		"BROKER GSTN",
		"BROKER GSTN STATE",
		"Total Inv Amt",
		"BROKERAGE Amount",
		"CGST @ 9%",
		"SGST @ 9%",
		"IGST",
		"GST TOTAL AMT",
		"DATE_FROM",
		"DATE_TO",
		"SAC Code",
	]
	for col in fill_columns:
		if not (result.get(col, "") or "").strip() and (page_fields.get(col, "") or "").strip():
			result[col] = page_fields[col]
	if not (result.get("Narration", "") or "").strip():
		result["Narration"] = "COMMISSION"
	return result


def process_pdf(path: Path, override_password: Optional[str], source_hint: Optional[Path] = None, source_display: Optional[str] = None) -> List[ReceiptLineItem]:
	pdf_bytes = path.read_bytes()
	pdf_digest = hashlib.sha1(pdf_bytes).hexdigest()
	rows: List[ReceiptLineItem] = []
	effective_source = source_hint or path
	effective_display = source_display or str(path)
	last_invoice_no = ""
	last_identity_fields: Dict[str, str] = {}
	last_invoice_basis: Dict[str, str] = {}
	is_bajaj_finance_file = _is_bajaj_finance_context(source_display or str(path))

	effective_password = override_password or infer_password_from_path(effective_source)
	page_texts = pdf_page_texts(pdf_bytes, str(path), effective_password)
	needs_ocr = any(len(text.strip()) < 20 for _, text in page_texts)
	force_ocr = is_jk_bank_source(effective_source) or is_monetary_ocr_priority_source(effective_source)

	if force_ocr or needs_ocr:
		if force_ocr:
			LOGGER.info("Forcing OCR for priority source: %s", path.name)
		LOGGER.info("Low text confidence in %s, switching to OCR for weak pages", path.name)
		target_pages = None if force_ocr else [page for page, text in page_texts if len(text.strip()) < 20]
		ocr_texts = {
			page: text
			for page, text in pdf_ocr_page_texts(
				pdf_bytes,
				prefer_google=True,
				force_google=force_ocr,
				source_id=effective_display,
				pdf_digest=pdf_digest,
				page_numbers=target_pages,
			)
		}
		merged: List[Tuple[int, str]] = []
		for page_num, text in page_texts:
			if force_ocr:
				merged_text = ocr_texts.get(page_num, "") or text
			else:
				merged_text = text if len(text.strip()) >= 20 else ocr_texts.get(page_num, "")
			merged.append((page_num, merged_text))
		page_texts = merged

	for page_num, page_text in page_texts:
		if not page_text.strip():
			continue
		
		is_axis = "axis" in effective_display.lower() and "bank" in effective_display.lower()
		page_ocr_score = score_ocr_text(page_text)
		low_conf_page = page_ocr_score < MIN_OCR_ALNUM_SCORE
		page_fields = {} if low_conf_page else extract_fields(page_text, is_axis_bank=is_axis)
		receipts = [page_text] if (is_india_post_source(effective_source) or low_conf_page) else split_receipts_from_page_text(page_text)
		page_rows: List[Dict[str, str]] = []
		ai_first = bool(get_azure_openai_config())
		ai_rows_first: List[Dict[str, str]] = []
		if ai_first:
			LOGGER.info("Running Azure mini extraction early for %s page %s", effective_display, page_num)
			ai_rows_first = extract_receipts_with_azure_llm(page_text, effective_display, page_num, _gst_mode_context_hint(page_rows) if page_rows else "")
			for ai_row in ai_rows_first:
				ai_row = merge_page_fallback_fields(ai_row, page_fields)
				ai_row = apply_party_overrides(ai_row, effective_source, page_text)
				ai_row = backfill_agent_pan(ai_row, page_text)
				ai_row = normalize_mapping_anomalies(ai_row, page_text, effective_source)
				ai_row = enforce_tax_mode_fields(ai_row)
				if not is_actual_receipt_row(ai_row, page_text):
					continue
				page_rows.append(ai_row)
		if not page_rows:
			for receipt_text in receipts:
				receipt_ocr_score = score_ocr_text(receipt_text)
				low_conf_receipt = receipt_ocr_score < MIN_OCR_ALNUM_SCORE
				if low_conf_receipt:
					continue
				fields = extract_fields(receipt_text, is_axis_bank=is_axis)
				fields = merge_page_fallback_fields(fields, page_fields)
				doc_context = f"{page_text} {receipt_text}"
				fields = apply_party_overrides(fields, effective_source, doc_context)
				fields = backfill_agent_pan(fields, doc_context)
				if (fields.get("Narration", "") or "").strip().upper() == "COMMISSION":
					inferred_narr = infer_narration_from_source_name(effective_display)
					if inferred_narr:
						fields["Narration"] = inferred_narr
				fields = normalize_mapping_anomalies(fields, doc_context, effective_source)
				fields = enforce_tax_mode_fields(fields)
				from_name = infer_invoice_no_from_source_name(effective_display)
				current_inv = (fields.get("Vendor Inv No", "") or "").strip()
				has_receipt_signal = bool(
					re.search(
						r"\b(invoice\s*(?:no\.?|number|reference\s*no)|bill\s*(?:no\.?|number)|document\s*no\.?)\b",
						receipt_text,
						re.IGNORECASE,
					)
				) or bool((fields.get("Vendor Inv Date", "") or "").strip()) or any(
					bool(clean_amount(fields.get(k, "")))
					for k in ["Total Inv Amt", "BROKERAGE Amount", "GST TOTAL AMT", "CGST @ 9%", "SGST @ 9%", "IGST"]
				)
			
				# If invoice number is still blank, try OCR fallback on the page
				if not current_inv and len(page_text.strip()) > 20:
					try:
						# Apply OCR to this specific page as fallback.
						image = render_pdf_page_image(pdf_bytes, page_num, dpi=300)
						if image is not None:
							ocr_text = ocr_image(image, prefer_google=True)
							if len(ocr_text.strip()) > 20:
								ocr_fields = extract_fields(ocr_text, is_axis_bank=is_axis)
								ocr_inv = (ocr_fields.get("Vendor Inv No", "") or "").strip()
								if ocr_inv and is_valid_invoice_no(ocr_inv):
									fields["Vendor Inv No"] = ocr_inv
									# Copy any other missing fields from OCR attempt
									for key in ocr_fields:
										if not fields.get(key, "").strip():
											fields[key] = ocr_fields[key]
						else:
							LOGGER.debug("OCR fallback produced no usable image for page %d", page_num)
					except Exception as e:
						LOGGER.debug("OCR fallback failed for page %d: %s", page_num, e)
			
				if from_name and has_receipt_signal:
					if not is_valid_invoice_no(current_inv):
						fields["Vendor Inv No"] = from_name
					elif from_name.isdigit() and 3 <= len(from_name) <= 6:
						# For files explicitly named as Invoice No <n>, prefer the filename token.
						if (not current_inv.isdigit()) or current_inv != from_name:
							fields["Vendor Inv No"] = from_name

				# Continuation-page rescue: if current chunk has monetary breakup but no invoice,
				# inherit invoice identity only when amounts match the latest invoice row.
				current_inv = (fields.get("Vendor Inv No", "") or "").strip()
				continuation_has_amount = any(
					bool(clean_amount(fields.get(k, "")))
					for k in ["Total Inv Amt", "BROKERAGE Amount", "GST TOTAL AMT", "CGST @ 9%", "SGST @ 9%", "IGST"]
				)
				if (not is_valid_invoice_no(current_inv)) and last_invoice_no and continuation_has_amount and _same_line_item_by_amount(fields, last_invoice_basis):
					fields["Vendor Inv No"] = last_invoice_no
					for col in [
						"Vendor Inv Date",
						"Agent Name",
						"Agent PAN",
						"Name of Service Receipient",
						"BALIC STATE",
						"BALIC GSTN",
						"BROKER GSTN STATE",
						"BROKER GSTN",
						"Narration",
						"DATE_FROM",
						"DATE_TO",
						"AGENT_CODE",
					]:
						if not (fields.get(col, "") or "").strip() and (last_identity_fields.get(col, "") or "").strip():
							fields[col] = last_identity_fields[col]

				# Do not create a row for page/chunk text that is not an actual receipt.
				if not is_actual_receipt_row(fields, receipt_text):
					continue
				matched_existing = None
				if (fields.get("Vendor Inv No", "") or "").strip():
					for existing in page_rows:
						if (existing.get("Vendor Inv No", "") or "").strip() == (fields.get("Vendor Inv No", "") or "").strip():
							matched_existing = existing
							break
				if matched_existing is not None:
					for col in OUTPUT_COLUMNS:
						if not (matched_existing.get(col, "") or "").strip() and (fields.get(col, "") or "").strip():
							matched_existing[col] = fields[col]
				else:
					page_rows.append(fields)
				inv_added = (fields.get("Vendor Inv No", "") or "").strip()
				if is_valid_invoice_no(inv_added):
					last_invoice_no = inv_added
					last_invoice_basis = dict(fields)
					for col in [
						"Vendor Inv Date",
						"Agent Name",
						"Agent PAN",
						"Name of Service Receipient",
						"BALIC STATE",
						"BALIC GSTN",
						"BROKER GSTN STATE",
						"BROKER GSTN",
						"Narration",
						"DATE_FROM",
						"DATE_TO",
						"AGENT_CODE",
					]:
						last_identity_fields[col] = (fields.get(col, "") or "").strip()

		gst_mode_conflict_trigger = any(_looks_gst_mode_ambiguous(r) for r in page_rows)

		# Low-cost AI fallback: call Azure mini only when deterministic extraction is weak.
		improbable_trigger = any(_looks_mapping_improbable(r) for r in page_rows)
		identity_conflict_trigger = any(_has_gstin_role_conflict(r) for r in page_rows)
		high_value_trigger = any(_is_high_value_row(r) for r in page_rows)
		should_try_ai = bool(get_azure_openai_config()) and (
			low_conf_page
			or
			not page_rows
			or any(_looks_mapping_incomplete(r) for r in page_rows)
			or improbable_trigger
			or identity_conflict_trigger
			or gst_mode_conflict_trigger
			or high_value_trigger
		)
		if ai_first:
			should_try_ai = False
		if should_try_ai:
			LOGGER.info("Running Azure mini extraction for %s page %s", effective_display, page_num)
			ai_rows = extract_receipts_with_azure_llm(page_text, effective_display, page_num, _gst_mode_context_hint(page_rows) if gst_mode_conflict_trigger else "")
			LOGGER.info("Azure mini returned %s receipt candidate(s) for %s page %s", len(ai_rows), effective_display, page_num)
			for ai_row in ai_rows:
				ai_row = merge_page_fallback_fields(ai_row, page_fields)
				ai_row = apply_party_overrides(ai_row, effective_source, page_text)
				ai_row = backfill_agent_pan(ai_row, page_text)
				ai_row = normalize_mapping_anomalies(ai_row, page_text, effective_source)
				ai_row = enforce_tax_mode_fields(ai_row)
				ai_inv = (ai_row.get("Vendor Inv No", "") or "").strip()
				ai_has_amount = any(
					bool(clean_amount(ai_row.get(k, "")))
					for k in ["Total Inv Amt", "BROKERAGE Amount", "GST TOTAL AMT", "CGST @ 9%", "SGST @ 9%", "IGST"]
				)
				if (not is_valid_invoice_no(ai_inv)) and last_invoice_no and ai_has_amount and _same_line_item_by_amount(ai_row, last_invoice_basis):
					ai_row["Vendor Inv No"] = last_invoice_no
					for col in [
						"Vendor Inv Date",
						"Agent Name",
						"Agent PAN",
						"Name of Service Receipient",
						"BALIC STATE",
						"BALIC GSTN",
						"BROKER GSTN STATE",
						"BROKER GSTN",
						"Narration",
						"DATE_FROM",
						"DATE_TO",
						"AGENT_CODE",
					]:
						if not (ai_row.get(col, "") or "").strip() and (last_identity_fields.get(col, "") or "").strip():
							ai_row[col] = last_identity_fields[col]
				if not is_actual_receipt_row(ai_row, page_text):
					continue

				# Merge by invoice number first to avoid duplicate rows for the same receipt.
				inv = (ai_row.get("Vendor Inv No", "") or "").strip()
				matched = None
				if inv:
					for existing in page_rows:
						if (existing.get("Vendor Inv No", "") or "").strip() == inv:
							matched = existing
							break
				if matched is not None:
					reconciled = _reconcile_regex_ai_monetary(matched, ai_row)
					for mcol in ["BROKERAGE Amount", "Total Inv Amt", "GST TOTAL AMT", "CGST @ 9%", "SGST @ 9%", "UTGST", "IGST"]:
						if (reconciled.get(mcol, "") or "").strip():
							matched[mcol] = reconciled[mcol]

					force_replace_identity_cols = {
						"BROKER GSTN",
						"BROKER GSTN STATE",
						"Agent PAN",
					}
					tax_mode_cols = {"CGST @ 9%", "SGST @ 9%", "UTGST", "IGST"}
					allow_identity_override = _should_ai_override_identity(matched, ai_row)
					matched_mode = _detect_clear_tax_mode(matched)
					ai_mode = _detect_clear_tax_mode(ai_row)
					for col in OUTPUT_COLUMNS:
						if allow_identity_override and col in force_replace_identity_cols and (ai_row.get(col, "") or "").strip():
							matched[col] = ai_row[col]
						elif col in tax_mode_cols and matched_mode in {"igst", "state"} and ai_mode in {"igst", "state"} and ai_mode != matched_mode:
							continue
						elif not (matched.get(col, "") or "").strip() and (ai_row.get(col, "") or "").strip():
							matched[col] = ai_row[col]
				else:
					page_rows.append(ai_row)
				inv_added = (ai_row.get("Vendor Inv No", "") or "").strip()
				if is_valid_invoice_no(inv_added):
					last_invoice_no = inv_added
					last_invoice_basis = dict(ai_row)
					for col in [
						"Vendor Inv Date",
						"Agent Name",
						"Agent PAN",
						"Name of Service Receipient",
						"BALIC STATE",
						"BALIC GSTN",
						"BROKER GSTN STATE",
						"BROKER GSTN",
						"Narration",
						"DATE_FROM",
						"DATE_TO",
						"AGENT_CODE",
					]:
						last_identity_fields[col] = (ai_row.get(col, "") or "").strip()

		if is_bajaj_finance_file:
			bajaj_total_rows = [r for r in page_rows if _is_bajaj_finance_total_row(page_text, r)]
			if bajaj_total_rows:
				best_bajaj_row = max(bajaj_total_rows, key=lambda r: (_monetary_signal_score(r), len([v for v in r.values() if str(v).strip()])))
				page_rows = [best_bajaj_row]
			elif page_rows:
				page_rows = [max(page_rows, key=lambda r: (_monetary_signal_score(r), len([v for v in r.values() if str(v).strip()]))) ]
			else:
				continue

		for fields in page_rows:
			fields = apply_confident_math_fill(fields)
			fields = enforce_tax_mode_fields(fields)
			fields = apply_gst_autofill(fields)
			if not (fields.get("Narration", "") or "").strip():
				inferred_narr = infer_narration_from_source_name(effective_display)
				fields["Narration"] = inferred_narr if inferred_narr else "COMMISSION"
			# Validate mathematical calculations and flag mismatches
			is_math_valid, math_reason = validate_math_extraction(fields)
			fields["Math Valid"] = "YES" if is_math_valid else f"NO: {math_reason}"
			fields["Missing Field and Why"] = build_missing_field_reason(fields)

			fields["Source File"] = effective_display
			fields["Source Page"] = str(page_num)
			rows.append(ReceiptLineItem(values=fields))

	# Remove repeated split artifacts within the same file while preserving order.
	seen: Set[Tuple[str, str, str, str, str, str, str, str, str]] = set()
	deduped: List[ReceiptLineItem] = []
	for item in rows:
		v = item.values
		key = (
			(v.get("Source Page", "") or "").strip(),
			(v.get("Vendor Inv No", "") or "").strip(),
			(v.get("Vendor Inv Date", "") or "").strip(),
			(v.get("Total Inv Amt", "") or "").strip(),
			(v.get("BROKERAGE Amount", "") or "").strip(),
			(v.get("GST TOTAL AMT", "") or "").strip(),
			(v.get("IGST", "") or "").strip(),
			(v.get("CGST @ 9%", "") or "").strip(),
			(v.get("SGST @ 9%", "") or "").strip(),
		)
		if key in seen:
			continue
		seen.add(key)
		deduped.append(item)
	rows = deduped
	return rows


def process_image(path: Path, source_hint: Optional[Path] = None, source_display: Optional[str] = None) -> List[ReceiptLineItem]:
	rows: List[ReceiptLineItem] = []
	effective_source = source_hint or path
	effective_display = source_display or str(path)
	is_bajaj_finance_file = _is_bajaj_finance_context(effective_display)
	for page_num, text in extract_text_from_image_file(path):
		fields = extract_fields(text)
		fields = apply_party_overrides(fields, effective_source, text)
		fields = backfill_agent_pan(fields, text)
		if (fields.get("Narration", "") or "").strip().upper() == "COMMISSION":
			inferred_narr = infer_narration_from_source_name(effective_display)
			if inferred_narr:
				fields["Narration"] = inferred_narr
		fields = normalize_mapping_anomalies(fields, text, effective_source)
		fields = enforce_tax_mode_fields(fields)
		fields = apply_gst_autofill(fields)
		if is_bajaj_finance_file and not _is_bajaj_finance_total_row(text, fields):
			continue
		if is_bajaj_finance_file and _is_bajaj_finance_row(fields) and _amount_to_float(fields.get("Total Inv Amt", "")) <= 0:
			continue
		if not (fields.get("Narration", "") or "").strip():
			inferred_narr = infer_narration_from_source_name(effective_display)
			fields["Narration"] = inferred_narr if inferred_narr else "COMMISSION"
		
		# Validate mathematical calculations and flag mismatches
		is_math_valid, math_reason = validate_math_extraction(fields)
		fields["Math Valid"] = "YES" if is_math_valid else f"NO: {math_reason}"
		fields["Missing Field and Why"] = build_missing_field_reason(fields)

		fields["Source File"] = effective_display
		fields["Source Page"] = str(page_num)
		rows.append(ReceiptLineItem(values=fields))
	return rows


def process_path(path: Path, override_password: Optional[str], source_hint: Optional[Path] = None, source_display: Optional[str] = None) -> List[ReceiptLineItem]:
	suffix = path.suffix.lower()
	if suffix == ".pdf":
		return process_pdf(path, override_password, source_hint=source_hint, source_display=source_display)
	if suffix in {".jpg", ".jpeg", ".png"}:
		return process_image(path, source_hint=source_hint, source_display=source_display)
	if suffix in {".xlsx", ".xlsm", ".xlsb", ".xls"}:
		return process_spreadsheet(path, source_hint=source_hint, source_display=source_display)
	if suffix == ".zip":
		all_rows: List[ReceiptLineItem] = []
		extracted_root = extract_zip_to_temp(path)
		for nested in find_candidate_files(extracted_root):
			try:
				rel = nested.relative_to(extracted_root)
				display = f"{path}::{rel.as_posix()}"
				nested_rows = process_path(nested, override_password, source_hint=path, source_display=display)
				if nested_rows:
					all_rows.extend(nested_rows)
				else:
					all_rows.append(build_placeholder_row(display, "", "No identifiable receipt data extracted from document"))
			except Exception as exc:
				LOGGER.warning("File %s had extraction error: %s", nested, exc)
				display = f"{path}::{nested.name}"
				all_rows.append(build_placeholder_row(display, "", f"Extraction error: {exc}"))
		return all_rows
	return []


def rows_to_dataframe(rows: List[ReceiptLineItem]) -> pd.DataFrame:
	values = [row.values for row in rows]
	df = pd.DataFrame(values)
	for column in OUTPUT_COLUMNS:
		if column not in df.columns:
			df[column] = ""

	# Merge continuation rows split across pages in the same source file.
	# Typical pattern: page 1 has invoice identity, page 2 has monetary breakup.
	if not df.empty:
		# Reconciliation pass: derive missing monetary fields from coherent components.
		for idx in df.index:
			row_dict = {col: ("" if pd.isna(df.at[idx, col]) else str(df.at[idx, col])) for col in OUTPUT_COLUMNS if col in df.columns}
			row_dict = enforce_tax_mode_fields(row_dict)
			row_dict = _maybe_correct_brokerage_from_total_and_tax(row_dict)
			row_dict = enforce_tax_mode_fields(row_dict)
			row_dict = _maybe_fix_ujjivan_gst_total(row_dict)
			row_dict = enforce_tax_mode_fields(row_dict)
			row_dict = _maybe_fix_city_union_gst_total(row_dict)
			row_dict = enforce_tax_mode_fields(row_dict)
			row_dict = _maybe_fix_dhanlaxmi_tax_components(row_dict)
			row_dict = enforce_tax_mode_fields(row_dict)
			row_dict = apply_confident_math_fill(row_dict)
			row_dict = enforce_tax_mode_fields(row_dict)
			row_dict = apply_gst_autofill(row_dict)
			brok = _amount_to_float(row_dict.get("BROKERAGE Amount", ""))
			igst = _amount_to_float(row_dict.get("IGST", ""))
			cgst = _amount_to_float(row_dict.get("CGST @ 9%", ""))
			sgst = _amount_to_float(row_dict.get("SGST @ 9%", ""))
			utgst = _amount_to_float(row_dict.get("UTGST", ""))
			gst_total = _amount_to_float(row_dict.get("GST TOTAL AMT", ""))
			total = _amount_to_float(row_dict.get("Total Inv Amt", ""))
			tax_component = igst if igst > 0 else (cgst + max(sgst, utgst))
			if gst_total <= 0 and tax_component > 0:
				row_dict["GST TOTAL AMT"] = f"{tax_component:.2f}".rstrip("0").rstrip(".")
				gst_total = tax_component
			if total <= 0 and brok > 0 and gst_total > 0:
				row_dict["Total Inv Amt"] = f"{(brok + gst_total):.2f}".rstrip("0").rstrip(".")
			if brok <= 0 and total > 0 and gst_total > 0 and total > gst_total:
				row_dict["BROKERAGE Amount"] = f"{(total - gst_total):.2f}".rstrip("0").rstrip(".")
			for col in ["BROKERAGE Amount", "CGST @ 9%", "SGST @ 9%", "UTGST", "IGST", "GST TOTAL AMT", "Total Inv Amt"]:
				if col in df.columns:
					df.at[idx, col] = row_dict.get(col, "")
		df["_src_file"] = df["Source File"].fillna("").astype(str).str.strip()
		df["_src_page_num"] = pd.to_numeric(df["Source Page"], errors="coerce").fillna(0).astype(int)
		df["_inv"] = df["Vendor Inv No"].fillna("").astype(str).str.strip()
		total_num = pd.to_numeric(df["Total Inv Amt"], errors="coerce")
		brok_num = pd.to_numeric(df["BROKERAGE Amount"], errors="coerce")
		gst_num = pd.to_numeric(df["GST TOTAL AMT"], errors="coerce")
		cgst_num = pd.to_numeric(df["CGST @ 9%"], errors="coerce").fillna(0.0)
		sgst_num = pd.to_numeric(df["SGST @ 9%"], errors="coerce").fillna(0.0)
		utgst_num = pd.to_numeric(df["UTGST"], errors="coerce").fillna(0.0)
		igst_num = pd.to_numeric(df["IGST"], errors="coerce").fillna(0.0)
		tax_num = gst_num.fillna(0.0) + cgst_num + sgst_num + utgst_num + igst_num
		weak_identity_row = df["_inv"].ne("") & (brok_num.fillna(0.0) <= 0.0) & (tax_num <= 0.0)
		strong_monetary_row = df["_inv"].eq("") & (brok_num.fillna(0.0) > 0.0) & (tax_num > 0.0)

		drop_idx: Set[int] = set()
		for src_file, src_group in df.groupby("_src_file", sort=False):
			if not src_file:
				continue
			group_idx = src_group.index.tolist()
			inv_idxs = [i for i in group_idx if bool(weak_identity_row.get(i, False))]
			monetary_idxs = [i for i in group_idx if bool(strong_monetary_row.get(i, False))]
			for inv_idx in inv_idxs:
				inv_total = total_num.get(inv_idx)
				best_idx = None
				best_score = -1.0
				for cand_idx in monetary_idxs:
					if cand_idx in drop_idx:
						continue
					cand_total = total_num.get(cand_idx)
					if pd.notna(inv_total) and pd.notna(cand_total) and abs(float(inv_total) - float(cand_total)) > 1.0:
						continue
					score = float(brok_num.get(cand_idx) or 0.0) + float(tax_num.get(cand_idx) or 0.0)
					if score > best_score:
						best_score = score
						best_idx = cand_idx
				if best_idx is None:
					continue

				# Keep stronger monetary row, backfill identity from page 1 row.
				for col in [
					"Vendor Inv No",
					"Vendor Inv Date",
					"Agent Name",
					"Agent PAN",
					"BALIC GSTN",
					"BALIC STATE",
					"BROKER GSTN",
					"BROKER GSTN STATE",
					"Name of Service Receipient",
					"Narration",
					"DATE_FROM",
					"DATE_TO",
					"AGENT_CODE",
				]:
					if not str(df.at[best_idx, col]).strip() and str(df.at[inv_idx, col]).strip():
						df.at[best_idx, col] = df.at[inv_idx, col]

				# Recompute row quality markers after merge.
				merged = {col: str(df.at[best_idx, col]) if col in df.columns else "" for col in OUTPUT_COLUMNS}
				is_math_valid, math_reason = validate_math_extraction(merged)
				df.at[best_idx, "Math Valid"] = "YES" if is_math_valid else f"NO: {math_reason}"
				df.at[best_idx, "Missing Field and Why"] = build_missing_field_reason(merged)

				drop_idx.add(inv_idx)

		if drop_idx:
			df = df.drop(index=list(drop_idx)).copy()

		# Remove weak non-invoice rows when a stronger same-total row exists in the same file.
		# This handles split-page cases where page 1 carries partial header values and page 2 has full details.
		total_num2 = pd.to_numeric(df["Total Inv Amt"], errors="coerce")
		brok_num2 = pd.to_numeric(df["BROKERAGE Amount"], errors="coerce")
		gst_num2 = pd.to_numeric(df["GST TOTAL AMT"], errors="coerce")
		inv_present2 = df["Vendor Inv No"].fillna("").astype(str).str.strip().ne("")
		drop_weak_idx: Set[int] = set()
		for src_file, src_group in df.groupby("_src_file", sort=False):
			if not src_file:
				continue
			idxs = src_group.index.tolist()
			def _num_or_zero(series: pd.Series, idx: int) -> float:
				val = series.get(idx)
				if pd.isna(val):
					return 0.0
				try:
					return float(val)
				except (TypeError, ValueError):
					return 0.0
			strong_idxs = [
				i for i in idxs
				if (inv_present2.get(i, False) or (_num_or_zero(brok_num2, i) > 0 and _num_or_zero(gst_num2, i) > 0))
			]
			weak_idxs = [
				i for i in idxs
				if (not inv_present2.get(i, False)) and (_num_or_zero(brok_num2, i) <= 0 or _num_or_zero(gst_num2, i) <= 0)
			]
			for weak_idx in weak_idxs:
				weak_total = total_num2.get(weak_idx)
				if pd.isna(weak_total) or float(weak_total) <= 0:
					continue
				weak_page = int(df.at[weak_idx, "_src_page_num"]) if "_src_page_num" in df.columns else 0
				for strong_idx in strong_idxs:
					if strong_idx == weak_idx:
						continue
					strong_total = total_num2.get(strong_idx)
					if pd.isna(strong_total):
						continue
					if abs(float(weak_total) - float(strong_total)) > 1.0:
						continue
					strong_page = int(df.at[strong_idx, "_src_page_num"]) if "_src_page_num" in df.columns else 0
					if strong_page >= weak_page:
						drop_weak_idx.add(weak_idx)
						break

		if drop_weak_idx:
			df = df.drop(index=list(drop_weak_idx)).copy()

		df = df.drop(columns=["_src_file", "_src_page_num", "_inv"])

	# Drop non-invoice placeholder rows that contain no extractable data.
	# These are support files/certificates/decrypt failures and inflate blanks/math failures.
	if not df.empty:
		reason = df["Missing Field and Why"].fillna("").astype(str)
		is_placeholder_reason = reason.str.contains(
			r"No identifiable receipt data extracted from document|Extraction failed: Unable to decrypt PDF",
			case=False,
			regex=True,
		)
		has_core_values = (
			df["Vendor Inv No"].fillna("").astype(str).str.strip().ne("")
			| df["Vendor Inv Date"].fillna("").astype(str).str.strip().ne("")
			| df["Total Inv Amt"].fillna("").astype(str).str.strip().ne("")
			| df["BROKERAGE Amount"].fillna("").astype(str).str.strip().ne("")
			| df["BALIC GSTN"].fillna("").astype(str).str.strip().ne("")
			| df["BROKER GSTN"].fillna("").astype(str).str.strip().ne("")
		)
		df = df.loc[~(is_placeholder_reason & ~has_core_values)].copy()
	# Collapse split artifacts where duplicate rows differ only by an empty Total Inv Amt.
	df["_source_file"] = df["Source File"].fillna("").astype(str).str.strip()
	df["_source_page"] = df["Source Page"].fillna("").astype(str).str.strip()
	df["_vendor_inv"] = df["Vendor Inv No"].fillna("").astype(str).str.strip()
	df["_vendor_date"] = df["Vendor Inv Date"].fillna("").astype(str).str.strip()
	df["_brok"] = df["BROKERAGE Amount"].fillna("").astype(str).str.strip()
	df["_gst"] = df["GST TOTAL AMT"].fillna("").astype(str).str.strip()
	df["_cgst"] = df["CGST @ 9%"].fillna("").astype(str).str.strip()
	df["_sgst"] = df["SGST @ 9%"].fillna("").astype(str).str.strip()
	df["_igst"] = df["IGST"].fillna("").astype(str).str.strip()
	df["_total"] = df["Total Inv Amt"].fillna("").astype(str).str.strip()
	df["_total_present"] = df["_total"].ne("")

	# Prefer rows where total is consistent with brokerage + GST.
	brok_num = pd.to_numeric(df["BROKERAGE Amount"], errors="coerce")
	total_num = pd.to_numeric(df["Total Inv Amt"], errors="coerce")
	cgst_num = pd.to_numeric(df["CGST @ 9%"], errors="coerce").fillna(0.0)
	sgst_num = pd.to_numeric(df["SGST @ 9%"], errors="coerce").fillna(0.0)
	utgst_num = pd.to_numeric(df["UTGST"], errors="coerce").fillna(0.0)
	igst_num = pd.to_numeric(df["IGST"], errors="coerce").fillna(0.0)
	state_tax_num = sgst_num.where(sgst_num >= utgst_num, utgst_num)
	gst_component_num = igst_num.where(igst_num > 0, cgst_num + state_tax_num)
	expected_total_num = brok_num + gst_component_num
	has_basis = brok_num.notna() & total_num.notna() & (gst_component_num > 0)
	df["_total_consistency_error"] = (total_num - expected_total_num).abs()
	# Penalize rows with missing GST components (tax, cgst, sgst, igst all empty)
	has_gst_component = (df["_cgst"].ne("") | df["_sgst"].ne("") | df["_igst"].ne(""))
	df.loc[~has_gst_component, "_total_consistency_error"] = 10**12
	df.loc[~has_basis, "_total_consistency_error"] = 10**12

	df = df.sort_values(["_total_present", "_total_consistency_error"], ascending=[False, True])
	df = df.drop_duplicates(
		subset=[
			"_source_file",
			"_vendor_inv",
			"_vendor_date",
		],
		keep="first",
	)
	df = df.drop(columns=[
		"_source_file",
		"_source_page",
		"_vendor_inv",
		"_vendor_date",
		"_brok",
		"_gst",
		"_cgst",
		"_sgst",
		"_igst",
		"_total",
		"_total_present",
		"_total_consistency_error",
	])

	# Prefer the most complete row when receipt splitting produces near-duplicate chunks.
	# Dedup key: Source File + Source Page + Total Inv Amt (uniquely identifies a receipt)
	# Keep the row with the most non-empty fields (highest _fill_score)
	score_columns = [c for c in OUTPUT_COLUMNS if c not in {"Source File", "Source Page"}]
	df["_fill_score"] = df[score_columns].fillna("").astype(str).apply(
		lambda row: sum(1 for v in row if str(v).strip()), axis=1
	)
	
	# Primary dedup key: source file + invoice number.
	# If invoice is missing, preserve distinct line items via page+monetary signature.
	vendor_inv_norm = df["Vendor Inv No"].astype(str).fillna("").str.strip()
	missing_inv_mask = vendor_inv_norm.eq("")
	fallback_sig = (
		df["Source Page"].astype(str).fillna("").str.strip()
		+ "|"
		+ df["Total Inv Amt"].astype(str).fillna("").str.strip()
		+ "|"
		+ df["BROKERAGE Amount"].astype(str).fillna("").str.strip()
		+ "|"
		+ df["GST TOTAL AMT"].astype(str).fillna("").str.strip()
		+ "|"
		+ df["CGST @ 9%"].astype(str).fillna("").str.strip()
		+ "|"
		+ df["SGST @ 9%"].astype(str).fillna("").str.strip()
		+ "|"
		+ df["IGST"].astype(str).fillna("").str.strip()
	)
	df["_dedup_key"] = (
		df["Source File"].astype(str).fillna("").str.strip()
		+ "|"
		+ vendor_inv_norm.where(~missing_inv_mask, "NOINV|" + fallback_sig)
		+ "|"
		+ df["Vendor Inv Date"].astype(str).fillna("").str.strip()
	)

	df = df.sort_values("_fill_score", ascending=False).drop_duplicates(subset=["_dedup_key"], keep="first")
	# Collapse exact duplicate receipts across different source files by business content.
	content_cols = [c for c in OUTPUT_COLUMNS if c not in {"Source File", "Source Page", "Math Valid", "Missing Field and Why"}]
	df["_content_key"] = df[content_cols].fillna("").astype(str).apply(
		lambda row: "|".join(_normalized_key_cell(value) for value in row),
		axis=1,
	)
	df = df.sort_values("_fill_score", ascending=False).drop_duplicates(subset=["_content_key"], keep="first")
	# Keep output ordered for review: same source file together, same agent together,
	# and page-wise ascending.
	df["_source_page_num"] = pd.to_numeric(df["Source Page"], errors="coerce").fillna(0).astype(int)
	df = df.sort_values(
		by=["Source File", "Agent Name", "_source_page_num", "Vendor Inv No"],
		ascending=[True, True, True, True],
		kind="mergesort",
	)
	df = df.drop(columns=["_fill_score", "_dedup_key", "_content_key"])
	if "_source_page_num" in df.columns:
		df = df.drop(columns=["_source_page_num"])

	# Final pass: always recompute quality flags after all monetary/tax reconciliation.
	if not df.empty:
		for idx in df.index:
			row_dict = {col: ("" if pd.isna(df.at[idx, col]) else str(df.at[idx, col])) for col in OUTPUT_COLUMNS if col in df.columns}
			is_math_valid, math_reason = validate_math_extraction(row_dict)
			df.at[idx, "Math Valid"] = "YES" if is_math_valid else f"NO: {math_reason}"
			df.at[idx, "Missing Field and Why"] = build_missing_field_reason(row_dict)
	return df[OUTPUT_COLUMNS]


def _normalized_key_cell(value: object) -> str:
	if value is None or (isinstance(value, float) and pd.isna(value)):
		return ""
	return str(value).strip().lower()


def _build_reconciliation_keys(df: pd.DataFrame, key_columns: Sequence[str]) -> pd.Series:
	if df.empty:
		return pd.Series(dtype="string")

	def join_key(row: pd.Series) -> str:
		return "|".join(_normalized_key_cell(row.get(col, "")) for col in key_columns)

	return df.apply(join_key, axis=1)


def write_audit_report(
	output_file: Path,
	df: pd.DataFrame,
	processed_sources: Sequence[str],
	baseline_output: Optional[Path] = None,
	audit_output: Optional[Path] = None,
) -> Optional[Path]:
	"""Write a non-blocking audit workbook for coverage and optional baseline reconciliation."""
	try:
		if audit_output is None:
			audit_output = output_file.with_name(f"{output_file.stem}_audit.xlsx")

		source_counts = (
			df.groupby("Source File", dropna=False)
			.size()
			.rename("rows_in_output")
			.reset_index()
			.rename(columns={"Source File": "source_file"})
		)
		source_counts["source_file"] = source_counts["source_file"].fillna("").astype(str)

		expected_df = pd.DataFrame({"source_file": [str(s) for s in processed_sources]})
		if not expected_df.empty:
			expected_df = expected_df.drop_duplicates(subset=["source_file"], keep="first")

		coverage = expected_df.merge(source_counts, on="source_file", how="left")
		if "rows_in_output" not in coverage.columns:
			coverage["rows_in_output"] = 0
		coverage["rows_in_output"] = coverage["rows_in_output"].fillna(0).astype(int)
		coverage["status"] = np.where(coverage["rows_in_output"] > 0, "HAS_ROWS", "NO_ROWS")

		summary_rows = [
			{"metric": "processed_sources", "value": int(len(expected_df))},
			{"metric": "sources_with_rows", "value": int((coverage["rows_in_output"] > 0).sum())},
			{"metric": "sources_with_no_rows", "value": int((coverage["rows_in_output"] == 0).sum())},
			{"metric": "output_rows", "value": int(len(df))},
		]

		with pd.ExcelWriter(audit_output, engine="openpyxl") as writer:
			pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="coverage_summary")
			coverage.sort_values(["status", "source_file"], ascending=[True, True]).to_excel(
				writer,
				index=False,
				sheet_name="source_coverage",
			)
			coverage.loc[coverage["rows_in_output"] == 0].sort_values("source_file").to_excel(
				writer,
				index=False,
				sheet_name="sources_with_no_rows",
			)

			if baseline_output and baseline_output.exists():
				baseline_df = pd.read_excel(baseline_output)
				for col in OUTPUT_COLUMNS:
					if col not in baseline_df.columns:
						baseline_df[col] = ""

				key_cols = ["Vendor Inv No", "Vendor Inv Date", "Total Inv Amt"]
				current_keyed = df.copy()
				baseline_keyed = baseline_df.copy()
				current_keyed["_recon_key"] = _build_reconciliation_keys(current_keyed, key_cols)
				baseline_keyed["_recon_key"] = _build_reconciliation_keys(baseline_keyed, key_cols)

				current_keys = set(current_keyed["_recon_key"].tolist())
				baseline_keys = set(baseline_keyed["_recon_key"].tolist())

				missing_keys = sorted(list(baseline_keys - current_keys))
				added_keys = sorted(list(current_keys - baseline_keys))

				missing_df = baseline_keyed[baseline_keyed["_recon_key"].isin(missing_keys)].copy()
				added_df = current_keyed[current_keyed["_recon_key"].isin(added_keys)].copy()
				missing_df = missing_df.drop(columns=["_recon_key"], errors="ignore")
				added_df = added_df.drop(columns=["_recon_key"], errors="ignore")

				recon_summary = pd.DataFrame(
					[
						{"metric": "baseline_rows", "value": int(len(baseline_df))},
						{"metric": "current_rows", "value": int(len(df))},
						{"metric": "baseline_unique_keys", "value": int(len(baseline_keys))},
						{"metric": "current_unique_keys", "value": int(len(current_keys))},
						{"metric": "missing_in_current", "value": int(len(missing_keys))},
						{"metric": "added_in_current", "value": int(len(added_keys))},
					],
				)
				recon_summary.to_excel(writer, index=False, sheet_name="reconciliation_summary")
				missing_df.to_excel(writer, index=False, sheet_name="missing_in_current")
				added_df.to_excel(writer, index=False, sheet_name="added_in_current")

		LOGGER.info("Wrote audit report to %s", audit_output)
		return audit_output
	except Exception as exc:  # pragma: no cover
		LOGGER.warning("Failed to write audit report: %s", exc)
		return None


def apply_agent_code_mapping_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply agent code mapping to all rows in the dataframe.
	
	For each row:
	- If AGENT_CODE is empty, try to find it via agent name similarity matching
	- Update Agent Name to the clean name from the mapping if a match is found
	"""
	if df.empty or "Agent Name" not in df.columns or "AGENT_CODE" not in df.columns:
		return df
	
	df = df.copy()
	
	for idx, row in df.iterrows():
		# Handle potential NaN values by converting to string first, then stripping
		agent_name_raw = row["Agent Name"]
		agent_name = (str(agent_name_raw) if pd.notna(agent_name_raw) else "").strip()
		
		agent_code_raw = row["AGENT_CODE"]
		agent_code = (str(agent_code_raw) if pd.notna(agent_code_raw) else "").strip()
		
		# Only process if code is missing but name is present
		if not agent_code and agent_name and agent_name != "nan":
			matched_code, clean_name, similarity = find_best_matching_agent_code(agent_name)
			if matched_code:
				df.at[idx, "AGENT_CODE"] = matched_code
				df.at[idx, "Agent Name"] = clean_name
				LOGGER.debug("Final mapping: '%s' -> code %s (similarity: %.2f)", agent_name, matched_code, similarity)
	
	return df


def run(
	input_path: Path,
	output_file: Path,
	password: Optional[str] = None,
	baseline_output: Optional[Path] = None,
	audit_output: Optional[Path] = None,
) -> Path:
	if not input_path.exists():
		raise FileNotFoundError(f"Input path does not exist: {input_path}")

	# Load agent code mappings from agentcode.xlsx
	load_agent_codes_from_xlsx()

	all_rows: List[ReceiptLineItem] = []
	processed_sources: List[str] = []
	for file_path in find_candidate_files(input_path):
		processed_sources.append(str(file_path))
		LOGGER.info("Processing: %s", file_path)
		try:
			rows = process_path(file_path, password)
			if rows:
				all_rows.extend(rows)
			else:
				all_rows.append(build_placeholder_row(str(file_path), "", "No identifiable receipt data extracted from document"))
		except Exception as exc:  # pragma: no cover
			LOGGER.exception("Failed to process %s: %s", file_path, exc)
			all_rows.append(build_placeholder_row(str(file_path), "", f"Extraction failed: {exc}"))

	df = rows_to_dataframe(all_rows)
	
	# Post-processing: Apply final agent code mapping to dataframe
	if not df.empty and AGENT_CODE_BY_NAME:
		df = apply_agent_code_mapping_to_dataframe(df)
	
	output_file.parent.mkdir(parents=True, exist_ok=True)
	df.to_excel(output_file, index=False)
	write_audit_report(
		output_file=output_file,
		df=df,
		processed_sources=processed_sources,
		baseline_output=baseline_output,
		audit_output=audit_output,
	)
	LOGGER.info("Wrote %s rows to %s", len(df), output_file)
	LOGGER.info(
		"Usage summary | google_vision_calls=%s | azure_ai_calls=%s | azure_ai_input_chars=%s | azure_ai_output_chars=%s",
		GOOGLE_VISION_CALL_COUNT,
		AZURE_AI_CALL_COUNT,
		AZURE_AI_INPUT_CHARS,
		AZURE_AI_OUTPUT_CHARS,
	)
	return output_file


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Extract receipt data from PDF/JPG/ZIP/Excel and write normalized Excel output.",
	)
	parser.add_argument("--input", required=True, help="Input file or folder path")
	parser.add_argument("--output", required=True, help="Output Excel file path (.xlsx)")
	parser.add_argument(
		"--password",
		required=False,
		default=None,
		help="Optional override password for encrypted PDFs",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable verbose logs",
	)
	parser.add_argument(
		"--baseline-output",
		required=False,
		default=None,
		help="Optional previous output Excel file for non-blocking reconciliation",
	)
	parser.add_argument(
		"--audit-output",
		required=False,
		default=None,
		help="Optional audit report Excel file path; defaults to <output>_audit.xlsx",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	configure_logging(args.verbose)
	run(
		Path(args.input),
		Path(args.output),
		args.password,
		Path(args.baseline_output) if args.baseline_output else None,
		Path(args.audit_output) if args.audit_output else None,
	)


if __name__ == "__main__":
	main()
