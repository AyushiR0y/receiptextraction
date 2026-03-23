import argparse
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

try:
	import pytesseract
except Exception:  # pragma: no cover
	pytesseract = None

try:
	from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover
	convert_from_bytes = None

try:
	from pypdf import PdfReader
except Exception:  # pragma: no cover
	PdfReader = None


LOGGER = logging.getLogger("receipt_extractor")


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


def find_first(patterns: Sequence[str], text: str, flags: int = re.IGNORECASE) -> str:
	for pattern in patterns:
		match = re.search(pattern, text, flags)
		if match:
			for group in match.groups():
				if group:
					return group.strip(" :-")
			return match.group(0).strip(" :-")
	return ""


def try_parse_date(raw: str) -> str:
	raw = raw.strip().replace(".", "/").replace("-", "/")
	match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", raw)
	return match.group(1) if match else ""


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


def ocr_image(image: Image.Image) -> str:
	if pytesseract is None:
		return ""
	prepared = preprocess_image_for_ocr(rotate_if_horizontal(image))
	try:
		text = pytesseract.image_to_string(prepared, config="--oem 3 --psm 6")
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
	invoice_no = find_first(
		[
			r"(?:vendor\s*inv(?:oice)?\s*(?:no|number)|invoice\s*(?:no|number))\s*[:\-]?\s*([A-Z0-9\-/]+)",
			r"\b(inv\d{3,}[A-Z0-9\-/]*)\b",
		],
		flattened,
	)
	vendor_date = try_parse_date(
		find_first(
			[
				r"(?:vendor\s*inv(?:oice)?\s*date|invoice\s*date)\s*[:\-]?\s*([0-9./\-]{6,12})",
			],
			flattened,
		)
	)

	amount_total = clean_amount(
		find_first(
			[
				r"(?:total\s*inv(?:oice)?\s*amt|total\s*invoice\s*amount|invoice\s*value|gross\s*amount)\s*[:\-]?\s*(\(?[0-9,]+(?:\.\d{1,2})?\)?)",
			],
			flattened,
		)
	)
	brokerage = clean_amount(
		find_first(
			[
				r"(?:brokerage\s*amount|brokerage)\s*[:\-]?\s*(\(?[0-9,]+(?:\.\d{1,2})?\)?)",
			],
			flattened,
		)
	)
	cgst = clean_amount(find_first([r"cgst\s*@?\s*9%?\s*[:\-]?\s*([0-9,]+(?:\.\d{1,2})?)"], flattened))
	sgst = clean_amount(find_first([r"sgst\s*@?\s*9%?\s*[:\-]?\s*([0-9,]+(?:\.\d{1,2})?)"], flattened))
	utgst = clean_amount(find_first([r"utgst\s*[:\-]?\s*([0-9,]+(?:\.\d{1,2})?)"], flattened))
	igst = clean_amount(find_first([r"igst\s*[:\-]?\s*([0-9,]+(?:\.\d{1,2})?)"], flattened))
	gst_total = clean_amount(
		find_first([r"(?:gst\s*total\s*amt|total\s*gst|tax\s*amount)\s*[:\-]?\s*([0-9,]+(?:\.\d{1,2})?)"], flattened)
	)
	agent_pan = find_first([r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"], flattened)
	balic_gstn = find_first([r"balic\s*gstn\s*[:\-]?\s*([0-9A-Z]{15})"], flattened)
	broker_gstn = find_first([r"broker\s*gstn\s*[:\-]?\s*([0-9A-Z]{15})"], flattened)
	sac_code = find_first([r"(?:sac\s*code|sac)\s*[:\-]?\s*([0-9]{4,8})"], flattened)
	date_from = try_parse_date(find_first([r"(?:date\s*from|period\s*from)\s*[:\-]?\s*([0-9./\-]{6,12})"], flattened))
	date_to = try_parse_date(find_first([r"(?:date\s*to|period\s*to)\s*[:\-]?\s*([0-9./\-]{6,12})"], flattened))

	row = {
		"AGENT_CODE": find_first([r"agent\s*code\s*[:\-]?\s*([A-Z0-9\-/]+)"], flattened),
		"Agent Name": find_first([r"agent\s*name\s*[:\-]?\s*([A-Z .,&()-]{3,})"], flattened),
		"Agent PAN": agent_pan,
		"Name of Service Receipient": find_first(
			[r"(?:name\s*of\s*service\s*rec(?:e|i)pient|service\s*recipient)\s*[:\-]?\s*([A-Z0-9 .,&()-]{3,})"],
			flattened,
		),
		"BALIC STATE": find_first([r"balic\s*state\s*[:\-]?\s*([A-Z ]{2,})"], flattened),
		"BALIC GSTN": balic_gstn,
		"BROKER GSTN STATE": find_first([r"broker\s*gstn\s*state\s*[:\-]?\s*([A-Z ]{2,})"], flattened),
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
		"Narration": find_first([r"(?:narration|description|remarks)\s*[:\-]?\s*([A-Z0-9 .,&()/\-]{5,})"], flattened),
		"Type": find_first([r"\btype\s*[:\-]?\s*([A-Z ]{3,})"], flattened),
		"Micro/Non Micro": find_first([r"(?:micro\s*/\s*non\s*micro|micro\s*non\s*micro)\s*[:\-]?\s*([A-Z ]{3,})"], flattened),
		"SAC Code": sac_code,
	}

	if not row["Type"]:
		row["Type"] = find_first([r"\btype\s*[:\-]?\s*([A-Z ]{3,})"], flattened)

	for col in OUTPUT_COLUMNS:
		row.setdefault(col, "")

	return row


def has_meaningful_data(row: Dict[str, str]) -> bool:
	key_fields = [
		"Vendor Inv No",
		"Vendor Inv Date",
		"Total Inv Amt",
		"BROKERAGE Amount",
		"GST TOTAL AMT",
		"Agent PAN",
		"BALIC GSTN",
		"BROKER GSTN",
		"SAC Code",
		"Narration",
	]
	return any((row.get(field, "") or "").strip() for field in key_fields)


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
		receipts = split_receipts_from_page_text(page_text)
		for receipt_text in receipts:
			fields = extract_fields(receipt_text)
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
