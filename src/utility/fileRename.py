from os import rename, listdir
SimNum = 20
prefix = "MCInputs"
fnames = [f for f in listdir('.') if f.startswith(prefix)]
for _, fname in enumerate(fnames):
    i = fname[-7:-4] 
    newname = fname[:-7] + '{:0>3d}'.format(int(i)+SimNum) +'.csv'
    print i, fname, newname
    rename(fname, newname)


prefix = "MCOutputs0"
fnames = [f for f in listdir('.') if f.startswith(prefix)]
for _, fname in enumerate(fnames):
    i = fname[-7:-4] 
    newname = fname[:-7] + '{:0>3d}'.format(int(i)+SimNum) +'.csv'
    print i, fname, newname
    rename(fname, newname)

prefix = "MCOutputs1"
fnames = [f for f in listdir('.') if f.startswith(prefix)]
for _, fname in enumerate(fnames):
    i = fname[-7:-4] 
    newname = fname[:-7] + '{:0>3d}'.format(int(i)+SimNum) +'.csv'
    print i, fname, newname
    rename(fname, newname)

prefix = "MCOutputs2"
fnames = [f for f in listdir('.') if f.startswith(prefix)]
for _, fname in enumerate(fnames):
    i = fname[-7:-4] 
    newname = fname[:-7] + '{:0>3d}'.format(int(i)+SimNum) +'.csv'
    print i, fname, newname
    rename(fname, newname)

