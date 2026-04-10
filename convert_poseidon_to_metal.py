#!/usr/bin/env python3
"""Convert Poseidon constants from decimal format to Metal nested hex format."""

def to_hex_limbs(dec_values):
    """Convert list of decimal uint32 values to Metal hex format."""
    return ", ".join("0x%08xu" % v for v in dec_values)

def parse_metal_array(content, start_line, end_line):
    """Parse Metal format array into list of limb values."""
    lines = content.split('\n')
    result = []
    for line in lines[start_line-1:end_line]:
        # Extract hex values like 0x12345678u
        import re
        hex_values = re.findall(r'0x([0-9a-f]+)u', line.lower())
        if hex_values:
            for hv in hex_values:
                result.append(int(hv, 16))
    return result

def generate_metal_mds_row(limbs, row_idx):
    """Generate a Metal MDS row from 8 uint32 limbs (already in Montgomery form)."""
    # limbs are already the 8 uint32 values for this element
    # Convert to hex strings
    hex_strs = ["0x%08xu" % v for v in limbs]
    return "    {" + ", ".join(hex_strs) + "}"

def generate_metal_rc_round(limbs):
    """Generate a Metal RC round entry from 24 uint32 limbs (3 state elements x 8 limbs each)."""
    # 3 elements, each with 8 limbs
    parts = []
    for i in range(3):
        elem_limbs = limbs[i*8:(i+1)*8]
        hex_strs = ["0x%08xu" % v for v in elem_limbs]
        parts.append("{" + ", ".join(hex_strs) + "}")
    return "    " + ", ".join(parts)

def main():
    # Read the script output
    with open('/Users/carnation/.claude/projects/-Users-carnation-Documents-Claude-zkMetal/8854e9b9-924c-4ab5-b8c6-0cf7a47138c5/tool-results/bumg6nkmn.txt', 'r') as f:
        script_output = f.read()

    # Parse the Metal constants from script output
    # Find PALLAS MDS section
    lines = script_output.split('\n')

    pallas_mds = []
    vesta_mds = []
    pallas_rc = []
    vesta_rc = []

    current_section = None

    for line in lines:
        line = line.strip()
        if '// METAL PALLAS MDS' in line:
            current_section = 'pallas_mds'
            continue
        elif '// METAL VESTA MDS' in line:
            current_section = 'vesta_mds'
            continue
        elif '// METAL PALLAS RC' in line:
            current_section = 'pallas_rc'
            continue
        elif '// METAL VESTA RC' in line:
            current_section = 'vesta_rc'
            continue
        elif line.startswith('//'):
            current_section = None
            continue

        if current_section is None:
            continue

        # Skip empty lines and comments
        if not line or line.startswith('//') or line.startswith('constant'):
            continue

        # Remove trailing commas and braces
        line = line.rstrip(',').rstrip(';').strip()
        if not line or line == '}' or line == '};':
            continue

        # Parse hex values
        import re
        hex_values = re.findall(r'0x([0-9a-f]+)u', line.lower())
        if not hex_values:
            continue

        int_values = [int(hv, 16) for hv in hex_values]

        if current_section == 'pallas_mds':
            pallas_mds.extend(int_values)
        elif current_section == 'vesta_mds':
            vesta_mds.extend(int_values)
        elif current_section == 'pallas_rc':
            pallas_rc.extend(int_values)
        elif current_section == 'vesta_rc':
            vesta_rc.extend(int_values)

    print("Pallas MDS total limbs:", len(pallas_mds))
    print("Vesta MDS total limbs:", len(vesta_mds))
    print("Pallas RC total limbs:", len(pallas_rc))
    print("Vesta RC total limbs:", len(vesta_rc))

    # Verify counts
    assert len(pallas_mds) == 3 * 3 * 8, "Pallas MDS should have 72 limbs (3x3x8)"
    assert len(vesta_mds) == 3 * 3 * 8, "Vesta MDS should have 72 limbs (3x3x8)"
    assert len(pallas_rc) == 55 * 3 * 8, "Pallas RC should have 1320 limbs (55x3x8)"
    assert len(vesta_rc) == 55 * 3 * 8, "Vesta RC should have 1320 limbs (55x3x8)"

    # Generate Metal PALLAS MDS
    print("\n// Pallas Fp Poseidon MDS matrix (Montgomery form)")
    print("constant PallasFp PALLAS_POS_MDS[3][3] = {")
    for row in range(3):
        row_limbs = pallas_mds[row*3*8:(row+1)*3*8]
        elements = []
        for elem in range(3):
            elem_limbs = row_limbs[elem*8:(elem+1)*8]
            hex_strs = ["0x%08xu" % v for v in elem_limbs]
            elements.append("{" + ", ".join(hex_strs) + "}")
        sep = "," if row < 2 else ""
        print("    {" + ", ".join(elements) + "}" + sep)
    print("};")

    # Generate Metal PALLAS RC
    print("\n// Pallas Fp Poseidon round constants (Montgomery form)")
    print("constant PallasFp PALLAS_POS_RC[55][3] = {")
    for rnd in range(55):
        rnd_limbs = pallas_rc[rnd*3*8:(rnd+1)*3*8]
        elements = []
        for elem in range(3):
            elem_limbs = rnd_limbs[elem*8:(elem+1)*8]
            hex_strs = ["0x%08xu" % v for v in elem_limbs]
            elements.append("{" + ", ".join(hex_strs) + "}")
        sep = "," if rnd < 54 else ""
        print("    {" + ", ".join(elements) + "}" + sep)
    print("};")

    # Generate Metal VESTA MDS
    print("\n// Vesta Fp Poseidon MDS matrix (Montgomery form)")
    print("constant VestaFp VESTA_POS_MDS[3][3] = {")
    for row in range(3):
        row_limbs = vesta_mds[row*3*8:(row+1)*3*8]
        elements = []
        for elem in range(3):
            elem_limbs = row_limbs[elem*8:(elem+1)*8]
            hex_strs = ["0x%08xu" % v for v in elem_limbs]
            elements.append("{" + ", ".join(hex_strs) + "}")
        sep = "," if row < 2 else ""
        print("    {" + ", ".join(elements) + "}" + sep)
    print("};")

    # Generate Metal VESTA RC
    print("\n// Vesta Fp Poseidon round constants (Montgomery form)")
    print("constant VestaFp VESTA_POS_RC[55][3] = {")
    for rnd in range(55):
        rnd_limbs = vesta_rc[rnd*3*8:(rnd+1)*3*8]
        elements = []
        for elem in range(3):
            elem_limbs = rnd_limbs[elem*8:(elem+1)*8]
            hex_strs = ["0x%08xu" % v for v in elem_limbs]
            elements.append("{" + ", ".join(hex_strs) + "}")
        sep = "," if rnd < 54 else ""
        print("    {" + ", ".join(elements) + "}" + sep)
    print("};")

if __name__ == '__main__':
    main()
