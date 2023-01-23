from pathlib import Path
import os
import time

p = "/media/Data/Datasets/2022_Perugia/wvn_output/split"
paths = [str(s) for s in Path(p).rglob("*.txt")]
perugia_root: str = "/media/Data/Datasets/2022_Perugia"

ls = []
for path in paths:
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            p = line.strip()
            ls.append(os.path.join(perugia_root, p))


import tarfile

archive = tarfile.open(os.path.join(perugia_root, "file.gz.tar"), "w:gz", compresslevel=1)
# archive = tarfile.open(os.path.join( perugia_root, "file.tar") , "w")


features = [str(x).split("/")[-1] for x in Path(ls[0]).parent.parent.joinpath("features").iterdir() if x.is_dir()]
st = time.time()
for j, path in enumerate(ls):
    # Add image
    archive.add(path, arcname=path.replace(perugia_root, ""))
    # Add supervision mask
    path_sm = path.replace("image", "supervision_mask")
    archive.add(path_sm, arcname=path_sm.replace(perugia_root, ""))

    # Add all the featues
    l = str(Path(path).parent.parent)
    timestamp = path.split("/")[-1][:-3]
    for f in features:
        for k in ["center", "graph", "seg"]:
            feature = str(Path(l).joinpath("features", f, k, timestamp + ".pt"))
            try:
                archive.add(feature, arcname=feature.replace(perugia_root, ""))
            except:
                pass

    print(j, time.time() - st)

archive.close()
