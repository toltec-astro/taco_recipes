

if __name__ == "__main__":
    import sys
    import shutil
    from astropy.table import Table
    fn = sys.argv[1]
    shutil.copy(fn, fn + ".cut_tones.bak")
    t = Table.read(fn, format='ascii.ecsv')

    t=t[t['f_centered'] < 0]
    n = len(t)
    t = t[-10:]
    print(f"number of tones remaining: {len(t)}")
    t.write(fn, format='ascii.ecsv', overwrite=True)
