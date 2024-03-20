template = """
#ObservingScript -Name  "goto_90.lmtot" -Author "servo" -Date "Wed Mar 20 16:09:46 EDT 2024"
Stay Telescope;  Telescope  -Stay Stay
SourceEl Source;  Source  -CoordSys Po -El[0] {elev_deg} -El[1] {elev_deg}
On -TScan  0 -NSamp 1 -NumRepeats 1 -NumScans 1
"""

if __name__ == '__main__':
    import numpy as np
    elev_values = np.arange(30, 91, 5)

    for elev_deg in elev_values:
        content = template.format(elev_deg=elev_deg)
        filename = f"./generated_goto_{elev_deg:.0f}.lmtot"
        with open(filename, "w") as fo:
            fo.write(content.strip())

