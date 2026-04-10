#!/usr/bin/env python3
"""Convert Poseidon constants from C hex format to Metal hex format.

C format:  {{0xULL, 0xULL, 0xULL, 0xULL}}  // 4 x uint64 per element
Metal format:  {0xU, 0xU, ...}  // 8 x uint32 per element

Each uint64 splits into 2 uint32 in little-endian order.
"""

import re

def parse_ull_line(line):
    """Parse a line containing hex ULL values and return list of uint32 values."""
    # Find all hex values like 0x7ea1fac679c7902aULL
    pattern = r'0x([0-9a-fA-F]+)ULL'
    matches = re.findall(pattern, line)
    result = []
    for m in matches:
        val = int(m, 16)
        # Split into 2 uint32 in little-endian order
        low = val & 0xFFFFFFFF
        high = (val >> 32) & 0xFFFFFFFF
        result.extend([low, high])
    return result

def generate_metal_mds(content, mds_start_line, pallas=True):
    """Generate Metal MDS array from C format content."""
    lines = content.split('\n')

    # Find the MDS section
    mds_lines = []
    in_mds = False
    brace_count = 0

    for i, line in enumerate(lines):
        if i+1 >= mds_start_line:
            if '{' in line:
                in_mds = True
            if in_mds:
                mds_lines.append(line.strip())
                brace_count += line.count('{')
                brace_count -= line.count('}')
                if brace_count == 0 and in_mds:
                    break

    # Parse all uint32 values
    all_limbs = []
    for l in mds_lines:
        limbs = parse_ull_line(l)
        all_limbs.extend(limbs)

    # Should have 3*3*8 = 72 limbs for MDS
    print(f"  Total limbs: {len(all_limbs)}")

    field = "PallasFp" if pallas else "VestaFp"
    array_name = "PALLAS_POS_MDS" if pallas else "VESTA_POS_MDS"

    print(f"constant {field} {array_name}[3][3] = {{")
    for row in range(3):
        row_values = []
        for elem in range(3):
            elem_limbs = all_limbs[(row*3 + elem)*8:(row*3 + elem + 1)*8]
            hex_strs = ["0x%08xu" % v for v in elem_limbs]
            row_values.append("{{{" + ", ".join(hex_strs) + "}}}")
        sep = "," if row < 2 else ""
        print("    " + ", ".join(row_values) + sep)
    print("};")

def generate_metal_rc(content, rc_start_line, pallas=True):
    """Generate Metal RC array from C format content."""
    lines = content.split('\n')

    # Find the RC section
    rc_lines = []
    in_rc = False
    brace_count = 0

    for i, line in enumerate(lines):
        if i+1 >= rc_start_line:
            if 'PA_POS_RC' in line or 'VE_POS_RC' in line:
                in_rc = True
            if in_rc:
                rc_lines.append(line.strip())
                brace_count += line.count('{')
                brace_count -= line.count('}')
                if brace_count == 0 and in_rc and len(rc_lines) > 5:
                    break

    # Parse all uint32 values
    all_limbs = []
    for l in rc_lines:
        limbs = parse_ull_line(l)
        all_limbs.extend(limbs)

    # Should have 55*3*8 = 1320 limbs for RC
    print(f"  Total limbs: {len(all_limbs)}")

    field = "PallasFp" if pallas else "VestaFp"
    array_name = "PALLAS_POS_RC" if pallas else "VESTA_POS_RC"

    print(f"constant {field} {array_name}[55][3] = {{")
    for rnd in range(55):
        rnd_values = []
        for elem in range(3):
            elem_limbs = all_limbs[(rnd*3 + elem)*8:(rnd*3 + elem + 1)*8]
            hex_strs = ["0x%08xu" % v for v in elem_limbs]
            rnd_values.append("{{{" + ", ".join(hex_strs) + "}}}")
        sep = "," if rnd < 54 else ""
        print("    " + ", ".join(rnd_values) + sep)
    print("};")

def main():
    # Read the script output
    with open('/Users/carnation/.claude/projects/-Users-carnation-Documents-Claude-zkMetal/8854e9b9-924c-4ab5-b8c6-0cf7a47138c5/tool-results/bumg6nkmn.txt', 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Find line numbers for each section
    pallas_mds_line = None
    pallas_rc_line = None
    vesta_mds_line = None
    vesta_rc_line = None

    for i, line in enumerate(lines):
        if '// C PALLAS MDS' in line:
            pallas_mds_line = i + 2  # Skip the array declaration line
        elif 'PA_POS_RC[55][3]' in line:
            pallas_rc_line = i + 1
        elif '// C VESTA MDS' in line:
            vesta_mds_line = i + 2
        elif 'VE_POS_RC[55][3]' in line:
            vesta_rc_line = i + 1

    print(f"Pallas MDS starts at line: {pallas_mds_line}")
    print(f"Pallas RC starts at line: {pallas_rc_line}")
    print(f"Vesta MDS starts at line: {vesta_mds_line}")
    print(f"Vesta RC starts at line: {vesta_rc_line}")

    print("\n// METAL PALLAS MDS")
    generate_metal_mds(content, pallas_mds_line, pallas=True)

    print("\n// METAL PALLAS RC")
    generate_metal_rc(content, pallas_rc_line, pallas=True)

    print("\n// METAL VESTA MDS")
    generate_metal_mds(content, vesta_mds_line, pallas=False)

    print("\n// METAL VESTA RC")
    generate_metal_rc(content, vesta_rc_line, pallas=False)

if __name__ == '__main__':
    main()
