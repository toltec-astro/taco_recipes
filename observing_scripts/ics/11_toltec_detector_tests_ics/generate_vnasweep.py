template = """
#ObservingScript -Name  "01_full_range_atten_grid.toltecot" -Author "lmtmc" -Date "Thu Feb 27 16:01:34 UTC 2020"
ToltecProposalId ToltecBackend;  ToltecBackend  -ProjectId commissioning_2024 -RoachArg[0] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachCmd null
ToltecObsGoal ToltecBackend;  ToltecBackend  -ObsGoal FullAttenGrid -RoachArg[0] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachCmd null
ToltecRoachVnaLoadAmpsMode ToltecBackend;  ToltecBackend  -RoachArg[0] LUT -RoachCmd vna_load_amps_mode -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null
ToltecRoachVnaLoad ToltecBackend;  ToltecBackend  -RoachArg[0] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachCmd vna_load
ToltecSubObsNumMode ToltecBackend;  ToltecBackend  -RoachCmd null -SubObsMode 1
{loop_items}
ToltecObsNumMode ToltecBackend;  ToltecBackend  -RoachCmd null -SubObsMode 0
"""
loop_item_template = """
ToltecRoachAttensSetDrive ToltecBackend;  ToltecBackend  -RoachArg[0] {a_drive} -RoachCmd attens_set_drive -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null
ToltecRoachAttensSetSense ToltecBackend;  ToltecBackend  -RoachArg[0] {a_sense} -RoachCmd attens_set_sense -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null
ToltecRoachVnaSweep ToltecBackend;  ToltecBackend  -RoachArg[0] null -RoachArg[10] null -RoachArg[11] null -RoachArg[12] null -RoachArg[13] null -RoachArg[14] null -RoachArg[15] null -RoachArg[1] null -RoachArg[2] null -RoachArg[3] null -RoachArg[4] null -RoachArg[5] null -RoachArg[6] null -RoachArg[7] null -RoachArg[8] null -RoachArg[9] null -RoachCmd vna_sweep
"""

data = {"ts_len": 5}

loop_items = []
for a_sense in range(30, -1, -2):
    for a_drive in range(20, -1, -10):
        data["a_sense"] = a_sense
        data["a_drive"] = a_drive
        loop_items.append(loop_item_template.format(**data).strip())


body = template.format(loop_items="\n".join(loop_items), **data)

filename = "./01_full_range_atten_grid_vnasweep.toltecot"
with open(filename, "w") as fo:
    fo.write(body.strip())
