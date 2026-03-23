import argparse
import difflib
import io
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

try:
	from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover
	RapidOCR = None

try:
	from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover
	convert_from_bytes = None

try:
	from pypdf import PdfReader
except Exception:  # pragma: no cover
	PdfReader = None


LOGGER = logging.getLogger("receipt_extractor")
RAPIDOCR_ENGINE = None
RAPIDOCR_INIT_FAILED = False


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
]


@dataclass
class ReceiptLineItem:
	values: Dict[str, str]


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


def clean_amount(value: str) -> str:
	value = value.replace(",", "").strip()
	match = re.search(r"-?\d+(?:\.\d{1,2})?", value)
	return match.group(0) if match else ""


INDIAN_STATES = [
	"Andhra Pradesh",
	"Arunachal Pradesh",
	"Assam",
	"Bihar",
	"Chhattisgarh",
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
	"West Bengal",
	"Andaman and Nicobar Islands",
	"Chandigarh",
	"Dadra and Nagar Haveli and Daman and Diu",
	"Delhi",
	"Jammu and Kashmir",
	"Ladakh",
	"Lakshadweep",
	"Puducherry",
]


def normalize_state_name(raw_state: str) -> str:
	if not raw_state:
		return ""
	candidate = re.sub(r"\s+", " ", raw_state).strip()
	if not candidate:
		return ""

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


def normalize_balic_company_name(raw_name: str) -> str:
	if not raw_name:
		return ""
	name = re.sub(r"\s+", " ", raw_name).strip()
	name_low = name.lower()
	letters_only = re.sub(r"[^a-z]", "", name_low)
	canon_letters = re.sub(r"[^a-z]", "", "bajaj allianz life insurance company ltd")

	if "bajaj" in letters_only:
		close = difflib.get_close_matches(letters_only, [canon_letters], n=1, cutoff=0.48)
		if close:
			return "Bajaj Allianz Life Insurance Company Ltd"

	if "bajaj" in name_low and (("allianz" in name_low) or ("aianz" in name_low) or ("amlanz" in name_low) or ("alianz" in name_low)):
		return "Bajaj Allianz Life Insurance Company Ltd"

	return name


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
	candidate = (value or "").strip()
	if len(candidate) < 4:
		return False
	if not re.search(r"\d", candidate):
		return False
	if candidate.lower() in {"date", "particulars", "invoice", "inv"}:
		return False
	return True


def find_first(patterns: Sequence[str], text: str, flags: int = re.IGNORECASE) -> str:
	for pattern in patterns:
		match = re.search(pattern, text, flags)
		if match:
			for group in match.groups():
				if group:
					return group.strip(" :-")
			return match.group(0).strip(" :-")
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
	scales = {"hundred": 100, "thousand": 1000, "lakh": 100000, "crore": 10000000}
	
	# Extract decimal part (paise) if present
	decimal_part = ""
	if "paise" in words:
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
	
	# Try to parse the main number
	current = 0
	result = 0
	
	for word in words.split():
		if word in ones:
			current += ones[word]
		elif word in tens:
			current += tens[word]
		elif word in scales:
			scale = scales[word]
			current *= scale
			if scale == 100:  # hundred
				result += current
				current = 0
			else:  # thousand, lakh, crore
				result += current
				current = 0
		elif word not in ["and", "only", "rupees"]:
			# Try to extract numbers if not a recognized word
			num_match = re.search(r"\d+", word)
			if num_match:
				current += int(num_match.group())
	
	result += current
	
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
	pattern = rf"{tax_name}[^%]{{0,220}}?(\d{{1,2}}(?:\.\d+)?)\s*%\s*([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)"
	match = re.search(pattern, processed, re.IGNORECASE | re.DOTALL)
	if match:
		candidate = clean_amount(match.group(2))
		if candidate and float(candidate) > 10:
			return candidate

	# 2) Segment scan: find % AMOUNT pairs near tax name
	tax_match = re.search(rf"{tax_name}", processed, re.IGNORECASE)
	if tax_match:
		start = tax_match.start()
		search_text = processed[start:start + 420]
		pairs = re.findall(r"\d{1,2}(?:\.\d+)?\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", search_text, re.IGNORECASE)
		for amount in pairs:
			candidate = clean_amount(amount)
			if candidate and float(candidate) > 10:
				return candidate

	# 3) Explicit label fallback: CGST amount 123.45
	label_match = re.search(rf"{tax_name}\s*(?:amount)?\s*[:\-]?\s*([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)", processed, re.IGNORECASE)
	if label_match:
		candidate = clean_amount(label_match.group(1))
		if candidate and float(candidate) > 10:
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


def try_parse_date(raw: str) -> str:
	raw = raw.strip().replace(".", "/").replace("-", "/")
	match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", raw)
	if not match:
		return ""
	
	date_str = match.group(1)
	parts = date_str.split("/")
	if len(parts) != 3:
		return ""
	
	try:
		# Try to parse as day/month/year or month/day/year
		day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
		
		# Fix 2-digit years: 00-30 -> 2000-2030, 31-99 -> 1931-1999
		if year < 100:
			year = 2000 + year if year <= 30 else 1900 + year
		
		# Validate ranges
		if not (1 <= month <= 12):
			return ""
		if not (1 <= day <= 31):
			return ""
		if not (1900 <= year <= 2100):
			return ""
		
		# More strict validation: check if day is reasonable for the month
		if month in [4, 6, 9, 11] and day > 30:
			return ""
		if month == 2 and day > 29:
			return ""
		
		return date_str
	except (ValueError, IndexError):
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


def ocr_image(image: Image.Image) -> str:
	engine = get_rapidocr_engine()
	if engine is None:
		return ""
	prepared = preprocess_image_for_ocr(rotate_if_horizontal(image))
	try:
		image_array = np.array(prepared)
		result, _ = engine(image_array)
		if not result:
			return ""
		chunks = [line[1] for line in result if len(line) > 1 and line[1]]
		text = "\n".join(chunks)
	except Exception as exc:
		LOGGER.warning("OCR failed for image: %s", exc)
		return ""
	return normalize_text(text)


def infer_password_from_name(file_name: str) -> Optional[str]:
	lower_name = file_name.lower()
	for key, password in BANK_PASSWORDS.items():
		if key in lower_name:
			return password
	return None


def pdf_page_texts(pdf_bytes: bytes, source_name: str, override_password: Optional[str]) -> List[Tuple[int, str]]:
	if PdfReader is None:
		raise RuntimeError("Missing dependency: pypdf")

	reader = PdfReader(io.BytesIO(pdf_bytes))
	if reader.is_encrypted:
		passwords_to_try: List[str] = []
		if override_password:
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


def pdf_ocr_page_texts(pdf_bytes: bytes) -> List[Tuple[int, str]]:
	if convert_from_bytes is None:
		return []
	try:
		images = convert_from_bytes(pdf_bytes, dpi=300)
	except Exception as exc:
		LOGGER.warning("PDF to image conversion failed: %s", exc)
		return []
	page_texts: List[Tuple[int, str]] = []
	for i, image in enumerate(images, start=1):
		text = ocr_image(image)
		page_texts.append((i, text))
	return page_texts


def extract_zip_to_temp(zip_path: Path) -> Path:
	temp_dir = Path(tempfile.mkdtemp(prefix="receipt_zip_"))
	with ZipFile(zip_path, "r") as zip_ref:
		zip_ref.extractall(temp_dir)
	return temp_dir


def find_candidate_files(input_path: Path) -> Iterable[Path]:
	supported = {".pdf", ".jpg", ".jpeg", ".png", ".zip"}
	if input_path.is_file() and input_path.suffix.lower() in supported:
		yield input_path
		return

	for path in input_path.rglob("*"):
		if path.is_file() and path.suffix.lower() in supported:
			yield path


def extract_text_from_image_file(path: Path) -> List[Tuple[int, str]]:
	image = Image.open(path)
	return [(1, ocr_image(image))]


def split_receipts_from_page_text(page_text: str) -> List[str]:
	separators = [
		r"\n\s*(?:invoice|tax\s+invoice)\s*(?:no|number)?\b",
		r"\n\s*vendor\s+inv\s+no\b",
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
						if part.strip():
							new_chunks.append(part)
						continue
					prefixed = "Invoice No " + part
					if prefixed.strip():
						new_chunks.append(prefixed)
		chunks = new_chunks

	filtered = [c.strip() for c in chunks if len(c.strip()) > 30]
	return filtered if filtered else [page_text]


def extract_fields(text: str) -> Dict[str, str]:
	flattened = text.replace("\n", " ")
	flattened = re.sub(r"\s+", " ", flattened).strip()
	invoice_no = find_first(
		[
			r"(?:Ref\s*Invoice\s*No|vendor\s*inv(?:oice)?\s*(?:no|number)|invoice\s*(?:no|number))\s*\.?\s*[:\-]?\s*([A-Z0-9\-/]+)",
			r"\b(inv\d{3,}[A-Z0-9\-/]*)\b",
		],
		flattened,
	)
	if not is_valid_invoice_no(invoice_no):
		invoice_no = ""

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
				r"(?:taxable\s*value|total\s*inv(?:oice)?\s*amt|total\s*invoice\s*amount|invoice\s*value|gross\s*amount|imf\s*fees)\s*[:\-]?\s*(\(?[0-9,]+(?:\.\d{1,2})?\)?)",
			],
			flattened,
		)
	)

	# Prefer amount-in-words conversion when present to avoid OCR decimal corruption (e.g., 6700.04 -> 6704)
	amount_words_match = re.search(r"Amount\s+in\s+words?\s*[:-]?\s*([A-Za-z\s]+?)(?:\s+only|$)", flattened, re.IGNORECASE)
	if amount_words_match:
		words_text = amount_words_match.group(1)
		converted = words_to_number(words_text)
		if converted:
			amount_total = converted
	
	# Brokerage = Taxable Value (appears in the tax table)
	brokerage = clean_amount(
		find_first(
			[
				r"(?:Taxable\s+Value|brokerage\s*amount|commission\s*amount)\s*[:\-]?\s*(\(?[0-9,]+(?:\.\d{1,2})?\)?)",
			],
			flattened,
		)
	)

	# Commission-line fallback (common receipt layout)
	bundle_taxable, bundle_cgst, bundle_sgst, bundle_total = extract_tax_bundle_from_commission_line(flattened)
	if not brokerage and bundle_taxable:
		brokerage = bundle_taxable
	if (not amount_total) and bundle_total:
		amount_total = bundle_total
	
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

	# Final fallback: use first two percentage-amount pairs anywhere in text
	if (not cgst or float(cgst) <= 10) or (not sgst or float(sgst) <= 10):
		global_pairs = re.findall(r"\d{1,2}(?:\.\d+)?\s*%\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", flattened, re.IGNORECASE)
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
			elif "9%" in flattened or "9 %" in flattened:
				all_amounts = re.findall(r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)", flattened)
				cleaned = [clean_amount(a) for a in all_amounts if clean_amount(a)]
				cleaned = [a for a in cleaned if float(a) > 10]
				if len(cleaned) >= 2:
					cgst = cleaned[0]
					sgst = cleaned[1]
				elif len(cleaned) == 1:
					cgst = cleaned[0]
					sgst = cleaned[0]
	except (ValueError, TypeError):
		pass
	
	# UTGST and SGST are interchangeable (both for state-level tax)
	# Extract UTGST independently, but validate it's an amount not a percentage
	utgst = clean_amount(find_first([
		r"utgst\s*(?:@|at)?\s*(?:\d{1,2}(?:\.\d+)?\s*%)?\s*([0-9,]+(?:\.\d{1,2})?)"
	], flattened))
	# Reject if it's just a percentage (≤ 10)
	try:
		if utgst and float(utgst) <= 10:
			utgst = ""
	except (ValueError, TypeError):
		pass
	
	# IGST is for inter-state transactions; should NOT exist if SGST/UTGST exist
	# SGST, UTGST, and IGST are mutually exclusive:
	# - Intra-state transaction: SGST or UTGST (state-level, only one)
	# - Inter-state transaction: IGST (integrated, replaces SGST/UTGST)
	igst = ""
	
	# Only extract IGST if neither SGST nor UTGST was found
	if (not sgst or float(sgst) <= 10) and (not utgst or float(utgst) <= 10):
		igst = clean_amount(extract_tax_amount(flattened, "IGST"))
		if not igst:
			igst = clean_amount(find_first([
				r"igst\s*(?:@|at)?\s*([0-9,]+(?:\.\d{1,2})?)"
			], flattened))
	else:
		# If SGST or UTGST exists, ensure IGST is blank (they're mutually exclusive)
		igst = ""
	
	# GST TOTAL = CGST + SGST (intra-state) OR CGST + UTGST (UT intra-state) OR CGST + IGST (inter-state)
	# NOTE: GST TOTAL is ONLY calculated from component taxes (CGST+SGST/UTGST/IGST)
	# It is NOT extracted from PDF text to prevent overwriting with invoice totals
	# User should apply formula in Excel: M:M + N:N for final GST total
	gst_total = ""
	try:
		cgst_num = float(cgst.replace(",", "")) if cgst else 0
		sgst_num = float(sgst.replace(",", "")) if sgst else 0
		utgst_num = float(utgst.replace(",", "")) if utgst else 0
		igst_num = float(igst.replace(",", "")) if igst else 0
		
		# Prioritize SGST/UTGST (intra-state) over IGST
		state_tax = max(sgst_num, utgst_num)  # Get whichever is larger (one should be 0)
		if cgst_num + state_tax > 0:
			gst_total = f"{cgst_num + state_tax:.2f}"
		# Fall back to CGST + IGST for inter-state
		elif cgst_num + igst_num > 0:
			gst_total = f"{cgst_num + igst_num:.2f}"
	except (ValueError, AttributeError):
			pass
	agent_pan = find_first([r"(?:pan|p\.a\.n)\.?\s*[:\-]?\s*([A-Z]{5}[0-9]{4}[A-Z])\b|\b([A-Z]{5}[0-9]{4}[A-Z])\b"], flattened)
	if not agent_pan:
		agent_pan = find_first([r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"], flattened)

	# Try to capture GSTIN field (both as balic_gstn and broker_gstn from GSTIN field)
	# GSTIN format: 2 digits + 13 alphanumeric chars = 15 total, ending with check digit
	# Find all valid GSTINs and filter out incomplete ones
	all_gstins = re.findall(r"([0-9]{2}[A-Z0-9]{13})", flattened)
	# Filter to keep only legitimate GSTINs (exclude partial matches of CIN numbers)
	gstin_values = [g for g in all_gstins if re.match(r"[0-9]{2}[A-Z0-9]{11}[A-Z0-9]", g)]
	balic_gstn = gstin_values[0] if len(gstin_values) >= 1 else find_first([r"balic\s*gstn\s*[:\-]?\s*([0-9A-Z]{15})"], flattened)
	broker_gstn = gstin_values[-1] if len(gstin_values) > 1 else (gstin_values[0] if gstin_values else find_first([r"broker\s*gstn\s*[:\-]?\s*([0-9A-Z]{15})"], flattened))
	
	# Extract SAC code and date ranges from narration
	sac_code = find_first([r"(?:sac\s*code|sac|hsn\s*code|hsn)\s*[:\-]?\s*([0-9]{4,8})"], flattened)
	
	# Extract date range from narration "01 to 31 Oct'2025"
	date_from = ""
	date_to = ""
	date_range_match = re.search(r"(\d{2})\s+(?:to|through)\s+(\d{2})\s+([A-Za-z]{3,9})['\"]?([0-9]{4})", flattened)
	if date_range_match:
		day_from, day_to, month_name, year = date_range_match.groups()
		month_map = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
		            "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
		month_num = month_map.get(month_name[:3].lower(), "")
		if month_num:
			date_from = f"{day_from}/{month_num}/{year}"
			date_to = f"{day_to}/{month_num}/{year}"
	
	# Fallback to generic date range extraction
	if not date_from:
		date_from = try_parse_date(find_first([r"(?:date\s*from|period\s*from|from)\s*[:\-]?\s*([0-9./\-]{6,12})"], flattened))
	if not date_to:
		date_to = try_parse_date(find_first([r"(?:date\s*to|period\s*to|to)\s*[:\-]?\s*([0-9./\-]{6,12})"], flattened))

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

	agent_name = find_first(
		[
			r"(?:Bajaj\s+Housing\s+Finance|Bajaj\s+[A-Za-z\s&]+(?:Limited|Ltd))",
			r"agent\s*name\s*[:\-]?\s*([A-Z][A-Za-z .,&()-]{3,})",
		],
		flattened,
	)
	if not agent_name:
		agent_name = service_recipient
	if not agent_name and re.search(r"bajaj\s+life\s+insurance", flattened, re.IGNORECASE):
		agent_name = "Bajaj Allianz Life Insurance Company Ltd"

	agent_name = normalize_balic_company_name(agent_name)
	service_recipient = normalize_balic_company_name(service_recipient)
	if not service_recipient:
		service_recipient = agent_name
	if not service_recipient and re.search(r"bajaj\s+life\s+insurance", flattened, re.IGNORECASE):
		service_recipient = "Bajaj Allianz Life Insurance Company Ltd"

	balic_state = extract_state_from_text(flattened)
	broker_state_raw = find_first([
		r"broker\s*gstn\s*state\s*[:\-]?\s*([A-Z ]{2,})",
		r"broker\s*state\s*[:\-]?\s*([A-Za-z ]{2,40})",
	], flattened)
	broker_state = normalize_state_name(broker_state_raw)
	if not balic_state and broker_state:
		balic_state = broker_state

	row = {
		"AGENT_CODE": find_first([r"agent\s*code\s*[:\-]?\s*([A-Z0-9\-/]+)"], flattened),
		"Agent Name": agent_name,
		"Agent PAN": agent_pan,
		"Name of Service Receipient": service_recipient,
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
		"Narration": "COMMISSION",
		"Type": find_first([
			r"(?:commission|type)\s*[:\-]?\s*([A-Z][A-Z\s]{2,15}?)(?:\s*[_\-]|\s+Non|\s+Micro|$)",
			r"([A-Z][A-Z]{2,15})\s*(?:INDIVIDUAL|GROUP|CORPORATE)",
			r"\b(Individual|Group|Corporate)\b"],
			flattened),
		"Micro/Non Micro": find_first([
			r"(?:Micro|Non\s*Micro|NON-MICRO|MICRO)\s*(?:[_\-]\s*)?([A-Z][A-Z\s]{2,15})?",
			r"(Micro|Non[\\s-]*Micro)"],
			flattened),
		"SAC Code": sac_code,
	}

	if not row["Type"]:
		row["Type"] = find_first([r"\btype\s*[:\-]?\s*([A-Z ]{3,})"], flattened)

	for col in OUTPUT_COLUMNS:
		row.setdefault(col, "")

	return row


def has_meaningful_data(row: Dict[str, str]) -> bool:
	invoice_ok = is_valid_invoice_no(row.get("Vendor Inv No", ""))
	date_ok = bool((row.get("Vendor Inv Date", "") or "").strip())
	amount_ok = any(clean_amount(row.get(field, "")) for field in ["Total Inv Amt", "BROKERAGE Amount", "GST TOTAL AMT", "CGST @ 9%", "SGST @ 9%", "IGST"])
	pan_ok = bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", (row.get("Agent PAN", "") or "").strip(), re.IGNORECASE))
	balic_gstn_ok = bool(re.fullmatch(r"[0-9A-Z]{15}", (row.get("BALIC GSTN", "") or "").strip(), re.IGNORECASE))
	broker_gstn_ok = bool(re.fullmatch(r"[0-9A-Z]{15}", (row.get("BROKER GSTN", "") or "").strip(), re.IGNORECASE))
	sac_ok = bool(re.fullmatch(r"[0-9]{4,8}", (row.get("SAC Code", "") or "").strip()))

	if invoice_ok and (date_ok or amount_ok or pan_ok or balic_gstn_ok or broker_gstn_ok):
		return True

	signals = [date_ok, amount_ok, pan_ok, balic_gstn_ok, broker_gstn_ok, sac_ok]
	return sum(1 for flag in signals if flag) >= 2


def merge_page_fallback_fields(row: Dict[str, str], page_fields: Dict[str, str]) -> Dict[str, str]:
	result = dict(row)
	fill_columns = [
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


def process_pdf(path: Path, override_password: Optional[str]) -> List[ReceiptLineItem]:
	pdf_bytes = path.read_bytes()
	rows: List[ReceiptLineItem] = []

	page_texts = pdf_page_texts(pdf_bytes, path.name, override_password)
	needs_ocr = any(len(text.strip()) < 20 for _, text in page_texts)

	if needs_ocr:
		LOGGER.info("Low text confidence in %s, switching to OCR for weak pages", path.name)
		ocr_texts = {page: text for page, text in pdf_ocr_page_texts(pdf_bytes)}
		merged: List[Tuple[int, str]] = []
		for page_num, text in page_texts:
			merged_text = text if len(text.strip()) >= 20 else ocr_texts.get(page_num, "")
			merged.append((page_num, merged_text))
		page_texts = merged

	for page_num, page_text in page_texts:
		if not page_text.strip():
			continue
		page_fields = extract_fields(page_text)
		receipts = split_receipts_from_page_text(page_text)
		for receipt_text in receipts:
			fields = extract_fields(receipt_text)
			fields = merge_page_fallback_fields(fields, page_fields)
			if not has_meaningful_data(fields):
				continue
			fields["Source File"] = str(path)
			fields["Source Page"] = str(page_num)
			rows.append(ReceiptLineItem(values=fields))
	return rows


def process_image(path: Path) -> List[ReceiptLineItem]:
	rows: List[ReceiptLineItem] = []
	for page_num, text in extract_text_from_image_file(path):
		fields = extract_fields(text)
		if not has_meaningful_data(fields):
			continue
		fields["Source File"] = str(path)
		fields["Source Page"] = str(page_num)
		rows.append(ReceiptLineItem(values=fields))
	return rows


def process_path(path: Path, override_password: Optional[str]) -> List[ReceiptLineItem]:
	suffix = path.suffix.lower()
	if suffix == ".pdf":
		return process_pdf(path, override_password)
	if suffix in {".jpg", ".jpeg", ".png"}:
		return process_image(path)
	if suffix == ".zip":
		all_rows: List[ReceiptLineItem] = []
		extracted_root = extract_zip_to_temp(path)
		for nested in find_candidate_files(extracted_root):
			try:
				all_rows.extend(process_path(nested, override_password))
			except Exception as exc:
				LOGGER.warning("Skipping file %s due to error: %s", nested, exc)
		return all_rows
	return []


def rows_to_dataframe(rows: List[ReceiptLineItem]) -> pd.DataFrame:
	values = [row.values for row in rows]
	df = pd.DataFrame(values)
	for column in OUTPUT_COLUMNS:
		if column not in df.columns:
			df[column] = ""

	# Prefer the most complete row when receipt splitting produces near-duplicate chunks.
	# Dedup key: Source File + Source Page + Total Inv Amt (uniquely identifies a receipt)
	# Keep the row with the most non-empty fields (highest _fill_score)
	score_columns = [c for c in OUTPUT_COLUMNS if c not in {"Source File", "Source Page"}]
	df["_fill_score"] = df[score_columns].fillna("").astype(str).apply(
		lambda row: sum(1 for v in row if str(v).strip()), axis=1
	)
	
	# Simpler dedup key: just use Source File + Source Page + Total Inv Amt
	# This avoids issues where Agent PAN/BROKER GSTN may vary due to extraction inconsistencies
	# but the receipt is identical (same page, same amounts)
	df["_dedup_key"] = (
		df["Source File"].astype(str).fillna("")
		+ "|"
		+ df["Source Page"].astype(str).fillna("")
		+ "|"
		+ df["Total Inv Amt"].astype(str).fillna("")
	)

	df = df.sort_values("_fill_score", ascending=False).drop_duplicates(subset=["_dedup_key"], keep="first")
	df = df.drop(columns=["_fill_score", "_dedup_key"])
	return df[OUTPUT_COLUMNS]


def run(input_path: Path, output_file: Path, password: Optional[str] = None) -> Path:
	if not input_path.exists():
		raise FileNotFoundError(f"Input path does not exist: {input_path}")

	all_rows: List[ReceiptLineItem] = []
	for file_path in find_candidate_files(input_path):
		LOGGER.info("Processing: %s", file_path)
		try:
			all_rows.extend(process_path(file_path, password))
		except Exception as exc:  # pragma: no cover
			LOGGER.exception("Failed to process %s: %s", file_path, exc)

	df = rows_to_dataframe(all_rows)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	df.to_excel(output_file, index=False)
	LOGGER.info("Wrote %s rows to %s", len(df), output_file)
	return output_file


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Extract receipt data from PDF/JPG/ZIP and write normalized Excel output.",
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
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	configure_logging(args.verbose)
	run(Path(args.input), Path(args.output), args.password)


if __name__ == "__main__":
	main()
