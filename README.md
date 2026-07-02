# Transacs – Advanced Wiegand Card Utility

**Transacs** is a powerful, vendor-aware command-line tool for parsing, converting, validating, generating, and reverse-engineering Wiegand-format access control cards.

## How to Use

```bash
git clone https://github.com/axkurcom/transacs.git
cd transacs
# No dependencies beyond Python 3.8+
python3 transacs.py
# Run regression tests
python3 -m unittest
```

## Supported Wiegand Formats
| Bits | Name / Description                     | Facility Bits | Card Bits | Parity Positions          | Primary Vendors                  |
|------|----------------------------------------|---------------|-----------|---------------------------|----------------------------------|
| 26   | Standard 26-bit (H10301)               | 8             | 16        | Even(0), Odd(25)          | HID, Generic                     |
| 34   | HID 34-bit                             | 16            | 16        | Even(0), Odd(33)          | HID                              |
| 37   | HID Corporate 1000 (H10304)            | 16            | 19        | Even(0), Odd(36)          | HID                              |
| 33   | 33-bit HID-compatible without parity   | 8             | 25        | None                      | HID                              |
| 35   | 35-bit HID-compatible without parity   | 12            | 23        | None                      | HID                              |
| 36   | Generic 36-bit                         | 4             | 32        | None                      | Generic                          |
| 40   | Generic 40-bit                         | 16            | 24        | None                      | Generic                          |
| 42   | Generic 42-bit                         | 16            | 26        | None                      | Generic                          |
| 44   | Generic 44-bit                         |16             | 28        | None                      | Generic                          |
| 48   | Generic 48-bit                         |16             | 32        | None                      | Generic                          |
| 56   | Indala 56-bit                          |24             | 32        | None                      | Indala                           |
| 64   | Indala 64-bit                          |32             | 32        | None                      | Indala                           |

## Input Formats Accepted
- `Facility/Card` → e.g. `123/45678`
- Raw binary string → e.g. `010010101...`
- Full Wiegand binary with parity → e.g. `0110...01`
- Hexadecimal with `0x` prefix → `0x1A2B3C4D`
- Decimal (data bits or full) → `123456789`

The tool automatically strips/replaces `.` with `/` for compatibility with some readers.
Bare hex strings such as `ABCDEF` are not auto-detected as hex; use the `0x` prefix.

## Core Features

### 1. Forward Transformation
Converts `Facility/Card`, hex, decimal, or binary → complete Wiegand representation with:
- Correct even/odd parity insertion (for 26/34/37-bit)
- Decimal value of data bits
- Full hex string
- Full Wiegand bit string
- Vendor detection

### 2. Reverse Transformation (`--reverse`)
Takes raw binary, hex, or decimal of the **full** Wiegand stream (including parity bits) and extracts:
- Facility code
- Card number
- Validates parity
- Performs card validation

Requires `--format`; without it the command returns an explicit error.

### 3. Auto-Detection (`--detect` or no format specified)
Tries **all** supported formats and returns ranked results:
- Prioritizes formats with **valid parity**
- Then sorts by number of warnings/errors
- Shows top 3 recommended formats
- For `0x...` input, considers both data-value and full-stream interpretations when they differ, while suppressing duplicate results

### 4. Extended Card Validation & Knowledge Base
Built-in database of known cards and explicit ranges with automatic detection:
- HID test card range `0/1000`-`0/1001`
- HID Corporate 1000 patterns
- Indala common cards
- AWID facility patterns
- Generic reserved/test cards (0/0, 255/255, etc.)

Provides warnings for:
- Facility/Card = 0
- Duplicate cards in session
- Extremely high numbers
- Known test/admin cards

### 5. Test Card Generation (`--generate N`)
Generates random (or fixed-facility) cards with proper parity and validation.
Fixed `--facility` values are checked against the selected format before cards are generated.
When `--vendor` is provided, the selected format must be compatible with that vendor.

### 6. Batch Processing (`--batch`)
Processes a file line-by-line, skipping empty lines and preserving source line numbers in output.
Batch mode reuses the same handling as single-input mode, so it can be combined with `--format`, `--detect`, `--validate --format`, or `--reverse --format`.
Use `--output file` to write batch results to a file instead of stdout.

### 7. Interactive Mode
Run without arguments → drops into REPL with commands:
- `generate <count> <format> [facility]`
- `detect <input>`
- Normal card entry for instant conversion
- `exit` / `quit`

## Command-Line Options
| Option              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `input`             | Card number, hex, binary, or file path                                      |
| `--format N`        | Force specific bit format                                                   |
| `--generate N`      | Generate N random cards (requires `--format`)                               |
| `--facility N`      | Fix facility code when generating                                           |
| `--vendor <name>`   | Require generated format to be compatible with vendor                       |
| `--batch`           | Treat input as filename for batch processing                                |
| `--output file`     | Write batch results to file                                                 |
| `--reverse`         | Force reverse transformation (full Wiegand → FC/CN)                         |
| `--detect`          | Force auto-detection mode                                                   |
| `--validate`        | Only perform validation, no conversion                                     |
| `--verbose` / `-v`  | Enable debug logging and extra output                                       |

`--reverse` and `--validate` both require `input` and `--format`.
`--batch` requires `input` to be a file path and cannot be combined with `--generate`.

## Known Limitations & Potential Issues

| Issue                                                                 | Cause                                                                                 | Effect                                                                 |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| No parity validation on no-parity formats                            | Only 26/34/37 have configured parity positions; others are treated as "No parity"     | Vendor-specific/custom parity is not validated                          |
| Hex input without leading `0x` is treated as decimal                  | No fallback for bare hex strings                                                      | `ABCDEF` → treated as decimal, causes confusion                        |
| Auto-detection may suggest multiple valid formats                     | Several formats can produce same FC/CN with different bit layouts                    | User must manually verify which format the system actually uses       |
| No support for PIV, iCLASS, MIFARE DESFire, or other smart card formats | Strictly Wiegand magnetic-stripe style only                                           | Cannot handle modern high-security credentials                        |
| Duplicate detection is session-only                                   | `self.seen_cards` cleared on new run                                                  | Does not persist across invocations                                    |
| No CSV/JSON structured output                                         | Only pretty-printed text                                                              | Hard to pipe into other tools                                          |

## Example Usage with CLI Options

```text
# Standard conversion (auto-detect)
./transacs.py 201/12345

# Force 26-bit
./transacs.py 201/12345 --format 26

# Reverse from full hex (with parity)
./transacs.py 0x0A1B2C3D4E --reverse --format 26

# Generate 10 test cards for facility 50
./transacs.py --generate 10 --format 26 --facility 50

# Generate HID-compatible 26-bit test cards
./transacs.py --generate 10 --format 26 --vendor hid

# Auto-detect with details
./transacs.py 0100110011010101... --detect --verbose

# Validate a card with an explicit format
./transacs.py 201/12345 --validate --format 26

# Process one card per non-empty line
./transacs.py cards.txt --batch --format 26 --output results.txt
```

## Tests

```bash
python3 -m unittest
```

## Example Usage in Interactive
```text
root@debian:~# python3 transacs.py
 [ TRANSACS 1.0.0.0 ] 
Advanced Wiegand Card Utility - Interactive Mode
Supported formats: 26, 34, 37, 33, 35, 36, 40, 42, 44, 48, 56, 64
Vendors: hid, indala, awid, kantech, lenel, generic
Commands: detect, generate, validate, exit

Enter Card Number or Command > 123/45678
[SUCCESS] Format = 26-bit ✓
Facility    = 123
Card        = 45678
Decimal     = 8106606
Hex         = 0x07BB26E
Wiegand     = 10111101110110010011011101
Vendor      = GENERIC

Enter Card Number or Command > 
```
