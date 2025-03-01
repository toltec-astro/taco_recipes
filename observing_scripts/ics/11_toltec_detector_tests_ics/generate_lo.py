template = """
#ObservingScript -Name  "02_lo_grid_vnasweep.toltecot" -Author "lmtmc" -Date "Thu Feb 27 16:01:34 UTC 2020"
ToltecObsGoal ToltecBackend;  ToltecBackend  -ObsGoal LOGrid -RoachArg[0] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachCmd null
ToltecRoachVnaLoadAmpsMode ToltecBackend;  ToltecBackend  -RoachArg[0] LUT -RoachCmd vna_load_amps_mode -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null
ToltecRoachVnaLoad ToltecBackend;  ToltecBackend  -RoachArg[0] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachCmd vna_load
ToltecSubObsNumMode ToltecBackend;  ToltecBackend  -RoachCmd null -SubObsMode 1
{loop_items}
ToltecObsNumMode ToltecBackend;  ToltecBackend  -RoachCmd null -SubObsMode 0
"""
loop_item_template = """
ToltecRoachLoSetOffset ToltecBackend;  ToltecBackend  -RoachArg[0] {lo_offset_Hz} -RoachCmd lo_set_offset -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null
ToltecRoachVnaSweep ToltecBackend;  ToltecBackend  -RoachArg[0] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachCmd vna_sweep
"""

data = {}

loop_items = []
for lo_offset_kHz in range(-1000, 1000 + 1, 200):
    data["lo_offset_Hz"] = lo_offset_kHz * 1000
    loop_items.append(loop_item_template.format(**data).strip())


body = template.format(loop_items="\n".join(loop_items), **data)

filename = "./02_lo_grid_vnasweep.toltecot"
with open(filename, "w") as fo:
    fo.write(body.strip())
