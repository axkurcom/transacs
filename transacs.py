#!/usr/bin/env python3
"""
### Transacs - Advanced Wiegand Card Utility ###
"""

import sys
import random
import csv
import argparse
import logging
from typing import Optional, Tuple, List, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum

# Logging setup
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParityType(Enum):
    EVEN = "even"
    ODD = "odd"
    NONE = "none"

class AccessControlVendor(Enum):
    HID = "hid"
    INDALA = "indala"
    AWID = "awid"
    KANTECH = "kantech"
    LENEL = "lenel"
    GENERIC = "generic"

@dataclass
class WiegandConfig:
    """Wiegand format configuration"""
    bit_format: int
    fac_len: int
    card_len: int
    parity_config: Optional[Tuple[int, int, int, int]] = None
    description: str = ""
    supported_vendors: List[AccessControlVendor] = None
    
    def __post_init__(self):
        if self.supported_vendors is None:
            self.supported_vendors = [AccessControlVendor.GENERIC]

@dataclass
class CardValidationResult:
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    vendor: Optional[AccessControlVendor] = None

@dataclass
class TransformationResult:
    success: bool
    input_str: str
    format_used: Optional[int] = None
    facility: Optional[int] = None
    card: Optional[int] = None
    decimal_value: Optional[int] = None
    hex_value: Optional[str] = None
    wiegand_bits: Optional[str] = None
    parity_valid: bool = False
    validation_result: Optional[CardValidationResult] = None
    error_message: Optional[str] = None

class WiegandFormatManager:
    def __init__(self):
        self.formats: Dict[int, WiegandConfig] = self._initialize_formats()
        self.card_knowledge_base = self._initialize_card_knowledge_base()
    
    def _initialize_formats(self) -> Dict[int, WiegandConfig]:
        """Initialize Supported Formats"""
        return {
            26: WiegandConfig(26, 8, 16, (0, 12, 12, 24), 
                             "26-bit - Standard", [AccessControlVendor.HID, AccessControlVendor.GENERIC]),
            34: WiegandConfig(34, 16, 16, (0, 16, 16, 32), 
                             "34-bit - HID", [AccessControlVendor.HID]),
            37: WiegandConfig(37, 16, 20, (0, 18, 18, 36), 
                             "37-bit - HID Corporate 1000", [AccessControlVendor.HID]),
            
            33: WiegandConfig(33, 8, 25, None, "35-bit HID w/o parity", [AccessControlVendor.HID]),
            35: WiegandConfig(35, 12, 23, None, "35-bit - HID w/ parity", [AccessControlVendor.HID]),
            36: WiegandConfig(36, 4, 32, None, "36-bit", [AccessControlVendor.GENERIC]),
            
            40: WiegandConfig(40, 16, 24, None, "40-bit", [AccessControlVendor.GENERIC]),
            42: WiegandConfig(42, 16, 26, None, "42-bit", [AccessControlVendor.GENERIC]),
            44: WiegandConfig(44, 16, 28, None, "44-bit", [AccessControlVendor.GENERIC]),
            48: WiegandConfig(48, 16, 32, None, "48-bit", [AccessControlVendor.GENERIC]),
            56: WiegandConfig(56, 24, 32, None, "56-bit", [AccessControlVendor.INDALA]),
            64: WiegandConfig(64, 32, 32, None, "64-bit - Indala", [AccessControlVendor.INDALA]),
        }
    
    def _initialize_card_knowledge_base(self) -> Dict[Tuple[int, int], Dict]:
        """Card knowledge base with verified data"""
        return {
            # HID specific patterns
            (0, 1000): {"vendor": AccessControlVendor.HID, "note": "HID Test Card Range"},
            (0, 1001): {"vendor": AccessControlVendor.HID, "note": "HID Test Card"},
            (0, 65535): {"vendor": AccessControlVendor.HID, "note": "HID Max Facility"},
            (1, 100): {"vendor": AccessControlVendor.HID, "note": "Common HID Facility 1"},
            
            # Indala patterns
            (0, 50000): {"vendor": AccessControlVendor.INDALA, "note": "Indala Common Range"},
            (1000, 2000): {"vendor": AccessControlVendor.INDALA, "note": "Indala Facility 1000 Range"},
            
            # General access control patterns
            (0, 0): {"vendor": AccessControlVendor.GENERIC, "note": "System/Reserved Card"},
            (255, 255): {"vendor": AccessControlVendor.GENERIC, "note": "Test/Admin Card"},
            (12345, 67890): {"vendor": AccessControlVendor.GENERIC, "note": "Demo Card"},
            
            # AWID patterns
            (10, 100): {"vendor": AccessControlVendor.AWID, "note": "AWID Common Facility"},
            (10, 200): {"vendor": AccessControlVendor.AWID, "note": "AWID Test Card"},
        }
    
    def get_format(self, bit_format: int) -> Optional[WiegandConfig]:
        """Get format configuration"""
        return self.formats.get(bit_format)
    
    def get_supported_formats(self) -> List[int]:
        """Get list of supported formats"""
        return list(self.formats.keys())
    
    def get_formats_by_vendor(self, vendor: AccessControlVendor) -> List[WiegandConfig]:
        """Get formats for specific vendor"""
        return [config for config in self.formats.values() 
                if vendor in config.supported_vendors]
    
    def detect_card_type(self, facility: int, card: int) -> Optional[Dict]:
        """Detect card type based on knowledge base"""
        # Exact match
        key = (facility, card)
        if key in self.card_knowledge_base:
            return self.card_knowledge_base[key]
        
        # Range matching for facility
        for (fac_range, card_range), info in self.card_knowledge_base.items():
            if isinstance(fac_range, tuple) and isinstance(card_range, tuple):
                if fac_range[0] <= facility <= fac_range[1] and card_range[0] <= card <= card_range[1]:
                    return info
        
        return None

class WiegandTransformer:
    """Main Wiegand card transformer class"""
    
    def __init__(self):
        self.format_manager = WiegandFormatManager()
        self.seen_cards: Set[str] = set()
    
    def calculate_parity(self, bits: str, start: int, end: int, parity_type: ParityType) -> str:
        """Calculate parity bit for given range"""
        if start >= end or end > len(bits):
            return '0'
        
        count = sum(int(bit) for bit in bits[start:end])
        
        if parity_type == ParityType.EVEN:
            return '0' if count % 2 == 0 else '1'
        elif parity_type == ParityType.ODD:
            return '1' if count % 2 == 0 else '0'
        else:
            return '0'
    
    def parse_input(self, input_str: str, bit_format: int) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """Parse input string into facility, card and bit representation"""
        config = self.format_manager.get_format(bit_format)
        if not config:
            return None, None, None
        
        fac_len, card_len = config.fac_len, config.card_len
        data_len = fac_len + card_len
        input_str = input_str.strip().replace('.', '/')
        
        try:
            # Binary input with parity
            if all(c in '01' for c in input_str):
                if len(input_str) == bit_format and config.parity_config:
                    # Full Wiegand with parity - extract data bits
                    start_e, end_e, start_o, end_o = config.parity_config
                    even_data_length = end_e - start_e
                    odd_data_length = end_o - start_o
                    data_bits = input_str[1:1+even_data_length] + input_str[1+even_data_length+1:-1]
                elif len(input_str) == data_len:
                    # Data bits without parity
                    data_bits = input_str
                else:
                    return None, None, None
                    
                if len(data_bits) != data_len:
                    return None, None, None
                    
                facility = int(data_bits[:fac_len], 2)
                card = int(data_bits[fac_len:], 2)
                return facility, card, data_bits
            
            # Hex input
            if input_str.lower().startswith('0x'):
                hex_value = input_str[2:]
                if not hex_value:
                    return None, None, None
                
                decimal_value = int(hex_value, 16)
                # Try with full format first (with parity), then data only
                for target_len in [bit_format, data_len]:
                    data_bits = bin(decimal_value)[2:].zfill(target_len)
                    if len(data_bits) == target_len:
                        if target_len == bit_format and config.parity_config:
                            # Extract data bits from full Wiegand
                            start_e, end_e, start_o, end_o = config.parity_config
                            even_data_length = end_e - start_e
                            odd_data_length = end_o - start_o
                            data_bits = data_bits[1:1+even_data_length] + data_bits[1+even_data_length+1:-1]
                        
                        if len(data_bits) == data_len:
                            facility = int(data_bits[:fac_len], 2)
                            card = int(data_bits[fac_len:], 2)
                            return facility, card, data_bits
                return None, None, None
            
            # Decimal input
            if '/' not in input_str:
                decimal_value = int(input_str)
                # Try with full format first (with parity), then data only
                for target_len in [bit_format, data_len]:
                    data_bits = bin(decimal_value)[2:].zfill(target_len)
                    if len(data_bits) == target_len:
                        if target_len == bit_format and config.parity_config:
                            # Extract data bits from full Wiegand
                            start_e, end_e, start_o, end_o = config.parity_config
                            even_data_length = end_e - start_e
                            odd_data_length = end_o - start_o
                            data_bits = data_bits[1:1+even_data_length] + data_bits[1+even_data_length+1:-1]
                        
                        if len(data_bits) == data_len:
                            facility = int(data_bits[:fac_len], 2)
                            card = int(data_bits[fac_len:], 2)
                            return facility, card, data_bits
                return None, None, None
            
            # Facility/Card input
            parts = input_str.split('/')
            if len(parts) != 2:
                return None, None, None
            
            facility = int(parts[0])
            card = int(parts[1])
            
            max_fac = (1 << fac_len) - 1
            max_card = (1 << card_len) - 1
            
            if not (0 <= facility <= max_fac) or not (0 <= card <= max_card):
                return None, None, None
            
            fac_bits = bin(facility)[2:].zfill(fac_len)
            card_bits = bin(card)[2:].zfill(card_len)
            data_bits = fac_bits + card_bits
            
            return facility, card, data_bits
            
        except (ValueError, Exception):
            return None, None, None
    
    def add_parity(self, data_bits: str, bit_format: int) -> str:
        """Add parity bits to data"""
        config = self.format_manager.get_format(bit_format)
        if not config or not config.parity_config:
            return data_bits.zfill(bit_format)
        
        fac_len, card_len = config.fac_len, config.card_len
        expected_data_len = fac_len + card_len
        
        if len(data_bits) != expected_data_len:
            return data_bits.zfill(bit_format)
        
        start_e, end_e, start_o, end_o = config.parity_config
        
        # Calculate parity bits
        even_parity = self.calculate_parity(data_bits, start_e, end_e, ParityType.EVEN)
        odd_parity = self.calculate_parity(data_bits, start_o, end_o, ParityType.ODD)
        
        # Build full bit string with parity
        if bit_format == 26:
            return even_parity + data_bits + odd_parity
        elif bit_format == 34:
            return even_parity + data_bits + odd_parity
        elif bit_format == 37:
            return even_parity + data_bits[:18] + odd_parity + data_bits[18:]
        else:
            return even_parity + data_bits + odd_parity
    
    def check_parity(self, wiegand_bits: str, bit_format: int) -> Tuple[bool, str]:
        """Check parity bits correctness"""
        config = self.format_manager.get_format(bit_format)
        if not config or not config.parity_config:
            return True, "[INFO] No parity"
        
        if len(wiegand_bits) != bit_format:
            return False, f"[ERROR] Length Mismatch: expected {bit_format}, got {len(wiegand_bits)}"
        
        start_e, end_e, start_o, end_o = config.parity_config
        
        try:
            # Check even parity (usually first bit)
            even_data_length = end_e - start_e
            expected_even = self.calculate_parity(wiegand_bits[1:1+even_data_length], 
                                                0, even_data_length, ParityType.EVEN)
            if wiegand_bits[0] != expected_even:
                return False, "Even parity failed"
            
            # Check odd parity (usually last bit)
            odd_data_start = 1 + even_data_length
            odd_data_length = end_o - start_o
            expected_odd = self.calculate_parity(wiegand_bits[odd_data_start:odd_data_start+odd_data_length], 
                                              0, odd_data_length, ParityType.ODD)
            if wiegand_bits[-1] != expected_odd:
                return False, "[ERR] Odd parity failed"
            
            return True, "[INFO] Valid"
        
        except Exception as e:
            return False, f"[ERROR] Validation Incomplete: {str(e)}"
    
    def validate_card(self, facility: int, card: int, bit_format: int) -> CardValidationResult:
        """Extended access control card validation"""
        config = self.format_manager.get_format(bit_format)
        if not config:
            return CardValidationResult(False, [], ["[ERROR] Invalid Format"])
        
        fac_len, card_len = config.fac_len, config.card_len
        max_fac = (1 << fac_len) - 1
        max_card = (1 << card_len) - 1
        
        errors = []
        warnings = []
        
        # Basic range checks
        if facility < 0 or facility > max_fac:
            errors.append(f"[ERROR] Facility {facility} Out of Range: 0-{max_fac}")
        if card < 0 or card > max_card:
            errors.append(f"[ERROR] Card {card} Out of Range: 0-{max_card}")
        
        # Detect card type from knowledge base
        card_info = self.format_manager.detect_card_type(facility, card)
        vendor = card_info.get("vendor") if card_info else AccessControlVendor.GENERIC
        
        if card_info:
            note = card_info.get("note", "")
            if note:
                warnings.append(f"[INFO] Card type detected: {note}")
        
        # General warnings
        if facility == 0:
            warnings.append("Facility 0 may be Reserved for System")
        if card == 0:
            warnings.append("Card 0 may be Invalid")
        if facility > 65535:
            warnings.append("Very Long Facility Number")
        if card > 999999:
            warnings.append("Very Long Card Number")
        
        # Duplicate check
        card_key = f"{facility}/{card}/{bit_format}"
        if card_key in self.seen_cards:
            warnings.append("Duplicate card detected")
        else:
            self.seen_cards.add(card_key)
        
        return CardValidationResult(
            is_valid=len(errors) == 0,
            warnings=warnings,
            errors=errors,
            vendor=vendor
        )
    
    def transform_to_decimal(self, facility: int, card: int, data_bits: str, bit_format: int) -> TransformationResult:
        """Transform to various representation formats"""
        config = self.format_manager.get_format(bit_format)
        if not config:
            return TransformationResult(False, "", error_message="[ERROR] Invalid Format Transformation")
        
        fac_len, card_len = config.fac_len, config.card_len
        expected_len = fac_len + card_len
        
        if len(data_bits) != expected_len:
            return TransformationResult(False, "", error_message=f"[ERROR] Data Length Mismatch: expected {expected_len}, but got {len(data_bits)}")
        
        try:
            # Main transformations
            decimal_value = int(data_bits, 2)
            hex_value = hex(decimal_value)[2:].upper().zfill((bit_format + 3) // 4)
            wiegand_bits = self.add_parity(data_bits, bit_format)
            
            # Check parity
            parity_valid, parity_msg = self.check_parity(wiegand_bits, bit_format)
            
            # Card validation
            validation_result = self.validate_card(facility, card, bit_format)
            
            return TransformationResult(
                success=True,
                input_str=data_bits,
                format_used=bit_format,
                facility=facility,
                card=card,
                decimal_value=decimal_value,
                hex_value=hex_value,
                wiegand_bits=wiegand_bits,
                parity_valid=parity_valid,
                validation_result=validation_result
            )
            
        except Exception as e:
            return TransformationResult(False, "", error_message=f"[ERROR] Transformation Result: {str(e)}")
    
    def reverse_transform(self, input_str: str, bit_format: int) -> TransformationResult:
        """Reverse transformation from bits/hex/dec to facility and card"""
        try:
            config = self.format_manager.get_format(bit_format)
            if not config:
                return TransformationResult(False, input_str, error_message="[ERROR] Invalid Format Manager")
            
            # Convert input string to bit string
            input_str = input_str.strip()
            
            if input_str.startswith('0x'):
                decimal_value = int(input_str[2:], 16)
                full_bits = bin(decimal_value)[2:].zfill(bit_format)
            elif all(c in '01' for c in input_str):
                full_bits = input_str.zfill(bit_format)
            else:
                decimal_value = int(input_str)
                full_bits = bin(decimal_value)[2:].zfill(bit_format)
            
            if len(full_bits) != bit_format:
                return TransformationResult(False, input_str, 
                                          error_message=f"[ERROR] Expected {bit_format} bits, but got {len(full_bits)}")
            
            if not all(c in '01' for c in full_bits):
                return TransformationResult(False, input_str, error_message="[ERROR] Invalid Binary String")
            
            # Check parity if format has it
            parity_valid = True
            if config.parity_config:
                parity_valid, parity_msg = self.check_parity(full_bits, bit_format)
                if not parity_valid:
                    return TransformationResult(False, input_str, error_message=f"[ERROR] Parity: {parity_msg}")
            
            # Extract data
            fac_len, card_len = config.fac_len, config.card_len
            
            if config.parity_config:
                start_e, end_e, start_o, end_o = config.parity_config
                even_data_length = end_e - start_e
                odd_data_length = end_o - start_o
                data_bits = full_bits[1:1+even_data_length] + full_bits[1+even_data_length+1:-1]
            else:
                data_bits = full_bits
            
            if len(data_bits) != fac_len + card_len:
                return TransformationResult(False, input_str, 
                                          error_message=f"[ERROR] Data Length Mismatch: expected {fac_len + card_len}, but got {len(data_bits)}")
            
            facility = int(data_bits[:fac_len], 2)
            card = int(data_bits[fac_len:], 2)
            
            # Validate result
            validation_result = self.validate_card(facility, card, bit_format)
            
            return TransformationResult(
                success=True,
                input_str=input_str,
                format_used=bit_format,
                facility=facility,
                card=card,
                decimal_value=int(full_bits, 2),
                hex_value=hex(int(full_bits, 2))[2:].upper(),
                wiegand_bits=full_bits,
                parity_valid=parity_valid,
                validation_result=validation_result
            )
            
        except Exception as e:
            return TransformationResult(False, input_str, error_message=f"[ERROR] Reverse Transformation: {str(e)}")
    
    def auto_detect_format(self, input_str: str) -> List[TransformationResult]:
        """Auto-detect format from input"""
        results = []
        
        for bit_format in self.format_manager.get_supported_formats():
            # Direct transformation
            facility, card, data_bits = self.parse_input(input_str, bit_format)
            if data_bits is not None:
                result = self.transform_to_decimal(facility, card, data_bits, bit_format)
                if result.success:
                    results.append(result)
            
            # Reverse transformation (for binary/hex inputs)
            if input_str.startswith('0x') or all(c in '01' for c in input_str):
                reverse_result = self.reverse_transform(input_str, bit_format)
                if reverse_result.success:
                    results.append(reverse_result)
        
        # Sort: valid parity first, then by warning count
        results.sort(key=lambda x: (
            not x.parity_valid,
            len(x.validation_result.warnings) if x.validation_result else 0,
            len(x.validation_result.errors) if x.validation_result else 0
        ))
        
        return results
    
    def generate_test_cards(self, count: int, bit_format: int, 
                          vendor: Optional[AccessControlVendor] = None,
                          facility: Optional[int] = None) -> List[TransformationResult]:
        """Generate test cards"""
        config = self.format_manager.get_format(bit_format)
        if not config:
            return []
        
        results = []
        fac_len, card_len = config.fac_len, config.card_len
        
        for i in range(count):
            # Generate facility
            if facility is not None:
                fac_value = facility
            else:
                fac_value = random.randint(1, (1 << fac_len) - 2)  # Avoid 0 and max
            
            # Generate card
            card_value = random.randint(1, (1 << card_len) - 2)
            
            # Create bit representation
            fac_bits = bin(fac_value)[2:].zfill(fac_len)
            card_bits = bin(card_value)[2:].zfill(card_len)
            data_bits = fac_bits + card_bits
            
            # Transform
            result = self.transform_to_decimal(fac_value, card_value, data_bits, bit_format)
            results.append(result)
        
        return results
    
    def format_result_for_display(self, result: TransformationResult) -> str:
        """Format result for display"""
        if not result.success:
            return f"[ERROR] {result.error_message}"
        
        # Basic information
        lines = [
            f"[SUCCESS] Format = {result.format_used}-bit {'✓' if result.parity_valid else '⚠'}",
            f"Facility    = {result.facility}",
            f"Card        = {result.card}",
            f"Decimal     = {result.decimal_value}",
            f"Hex         = 0x{result.hex_value}",
            f"Wiegand     = {result.wiegand_bits}"
        ]
        
        # Vendor information
        if result.validation_result and result.validation_result.vendor:
            lines.append(f"Vendor      = {result.validation_result.vendor.value.upper()}")
        
        # Warnings and errors
        if result.validation_result:
            if result.validation_result.warnings:
                lines.append(f"[WARN] {', '.join(result.validation_result.warnings)}")
            if result.validation_result.errors:
                lines.append(f"[ERROR] {', '.join(result.validation_result.errors)}")
        
        return "\n".join(lines)

def main():
    """Main CLI function"""
    transformer = WiegandTransformer()
    parser = argparse.ArgumentParser(description="Advanced Wiegand Card Transformer with Access Control Systems Support")
    
    parser.add_argument('input', nargs='?', help="Input String or File")
    parser.add_argument('--format', type=int, help="Specify Wiegand Format")
    parser.add_argument('--generate', type=int, default=0, help="Generate N Random Cards")
    parser.add_argument('--facility', type=int, help="Fixed Facility for Generation")
    parser.add_argument('--vendor', type=str, help="Vendor for Generation/Validation")
    parser.add_argument('--batch', action='store_true', help="Treat Input as File for Batch Processing")
    parser.add_argument('--output', help="Output File for Batch Results")
    parser.add_argument('--reverse', action='store_true', help="Force Reverse Transformation")
    parser.add_argument('--detect', action='store_true', help="Auto-detect Format")
    parser.add_argument('--validate', action='store_true', help="Extended Validation only")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose Debug Mode")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Auto-detect format
    if args.detect and args.input:
        print("Auto-detecting Format...")
        results = transformer.auto_detect_format(args.input)
        
        if results:
            valid_results = [r for r in results if r.parity_valid and r.validation_result.is_valid]
            warning_results = [r for r in results if not r.parity_valid or not r.validation_result.is_valid]
            
            if valid_results:
                print("Recommended Formats - Valid Parity and Card:")
                for result in valid_results[:3]:  # Show top 3
                    print(f"- {result.format_used}-bit: {result.facility}/{result.card}")
            
            if warning_results and not valid_results:
                print("Compatible Formats - with Warn:")
                for result in warning_results[:3]:
                    status = "parity" if not result.parity_valid else "card"
                    print(f"- {result.format_used}-bit: {result.facility}/{result.card} - {status} issue")
        else:
            print("No Compatible Formats Found")
    
    # Generate cards
    elif args.generate > 0:
        if not args.format:
            print("[ERROR] '--format' Required for Generation")
            return
        
        vendor = AccessControlVendor(args.vendor) if args.vendor else None
        cards = transformer.generate_test_cards(args.generate, args.format, vendor, args.facility)
        
        print(f"Generated {len(cards)} Test Cards:")
        for i, card in enumerate(cards, 1):
            print(f"\nCard {i}:")
            print(transformer.format_result_for_display(card))
    
    # Reverse transformation
    elif args.reverse and args.input and args.format:
        result = transformer.reverse_transform(args.input, args.format)
        print(transformer.format_result_for_display(result))
    
    # Validation
    elif args.validate and args.input and args.format:
        facility, card, data_bits = transformer.parse_input(args.input, args.format)
        if facility is not None:
            validation = transformer.validate_card(facility, card, args.format)
            print(f"Validation for {facility}/{card} - {args.format}-bit:")
            print(f"Valid: {validation.is_valid}")
            if validation.vendor:
                print(f"Vendor: {validation.vendor.value}")
            if validation.warnings:
                print(f"Warn: {', '.join(validation.warnings)}")
            if validation.errors:
                print(f"Error: {', '.join(validation.errors)}")
        else:
            print("Cannot Parse Input for Validation")
    
    # Normal transformation
    elif args.input:
        if args.format:
            # Specific format
            facility, card, data_bits = transformer.parse_input(args.input, args.format)
            if data_bits is not None:
                result = transformer.transform_to_decimal(facility, card, data_bits, args.format)
                print(transformer.format_result_for_display(result))
            else:
                print(f"[ERROR] Cannot Parse Input as {args.format}-bit Format")
        else:
            # Auto-detect
            results = transformer.auto_detect_format(args.input)
            if results:
                best_result = results[0]
                print(transformer.format_result_for_display(best_result))
                
                if len(results) > 1 and args.verbose:
                    print(f"\nOther Compatible Formats - {len(results)-1}:")
                    for result in results[1:3]:  # Show 2 more options
                        print(f"- {result.format_used}-bit: {result.facility}/{result.card}")
            else:
                print("[ERROR] Cannot Parse Input with any Supported Format")
    
    # Interactive mode
    else:
        print(" [ TRANSACS 1.0.0.0] ")
        print("Advanced Wiegand Card Utility - Interactive Mode")
        print("Supported formats:", ", ".join(map(str, transformer.format_manager.get_supported_formats())))
        print("Vendors:", ", ".join([v.value for v in AccessControlVendor]))
        print("Commands: detect, generate, validate, exit")
        
        while True:
            try:
                user_input = input("\nEnter Card Number or Command > ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'detect':
                    print("Use: detect <card_number>")
                elif user_input.lower() == 'generate':
                    print("Use: generate <count> <format> <facility>")
                elif user_input.lower() == 'validate':
                    print("Use: validate <card_number> <format>")
                elif user_input.startswith('generate '):
                    parts = user_input.split()
                    if len(parts) >= 3:
                        count, fmt = int(parts[1]), int(parts[2])
                        facility = int(parts[3]) if len(parts) > 3 else None
                        cards = transformer.generate_test_cards(count, fmt, facility=facility)
                        for card in cards:
                            print(transformer.format_result_for_display(card))
                elif user_input.startswith('detect '):
                    card_input = user_input[7:]
                    results = transformer.auto_detect_format(card_input)
                    if results:
                        print(f"Found {len(results)} Compatible Formats")
                        for result in results[:3]:
                            print(transformer.format_result_for_display(result))
                    else:
                        print("No Compatible Formats Found")
                elif user_input:
                    # Normal transformation
                    results = transformer.auto_detect_format(user_input)
                    if results:
                        print(transformer.format_result_for_display(results[0]))
                    else:
                        print("[ERROR] Cannot Parse Input")
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                if args.verbose:
                    logger.error(f"[ERROR] {e}")
                print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
