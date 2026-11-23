"""Step 0: log each photo in ../my_inputs (name, original size, EXIF date/device,
source/license from ../my_inputs_sources.json), fix EXIF rotation, resize long side
to 1200 px (LANCZOS, aspect kept), save PNG to ../my_inputs_resized/. Originals untouched.
Run from ~/multinex_repro/Multinex."""
import os, json, re
from PIL import Image, ImageOps

SRC = '../my_inputs'; DST = '../my_inputs_resized'
os.makedirs(DST, exist_ok=True)
sources = {re.sub(r'[^A-Za-z0-9._-]+', '_', s['title'][5:]): s for s in json.load(open('../my_inputs_sources.json'))}
# short output names: the Commons titles are long
short = {'Zounohana': 'zounohana_park', 'Barely_Lit': 'barely_lit', 'Project_366': 'cobbled_lanes',
         'Budapest': 'budapest_street', '2014-12-20': 'sodium_streetlights'}
lines, meta = [], []
for fn in sorted(os.listdir(SRC)):
    im = Image.open(os.path.join(SRC, fn))
    ex = im.getexif()
    sub = ex.get_ifd(0x8769)
    date = sub.get(36867) or ex.get(306)  # DateTimeOriginal, else DateTime
    dev = ' '.join(str(v).strip('\x00 ') for v in (ex.get(271), ex.get(272)) if v) or None  # Make Model
    orient = ex.get(274, 1)
    w0, h0 = im.size
    im = ImageOps.exif_transpose(im).convert('RGB')
    w1, h1 = im.size
    s = 1200 / max(w1, h1)
    w2, h2 = round(w1 * s), round(h1 * s)
    out_name = next(v for k, v in short.items() if fn.startswith(k)) + '.png'
    im.resize((w2, h2), Image.LANCZOS).save(os.path.join(DST, out_name))
    src = sources[fn]
    meta.append(dict(file=out_name, original=fn, original_resolution=f'{w0}x{h0}', exif_orientation=orient,
                     resolution=f'{w2}x{h2}', exif_date=date, exif_device=dev,
                     source_page=src['page'], author=src['artist'], license=src['license']))
    lines.append(f'{fn}\n  original {w0}x{h0}, EXIF orientation {orient}, after rotation {w1}x{h1}, resized {w2}x{h2} -> {out_name}\n'
                 f'  EXIF date: {date}\n  EXIF device: {dev}\n  source: {src["page"]}\n  author: {src["artist"]}, license: {src["license"]}')
hdr = ('Photos: no own photos were provided; at the user\'s request these are phone photos downloaded from\n'
       'Wikimedia Commons (free licenses, originals with EXIF, SHA-1 checked: logs/10_download_photos.log).\n')
open('../logs/10_my_photos.txt', 'w').write(hdr + '\n'.join(lines) + '\n')
json.dump(meta, open('../logs/10_my_photos_meta.json', 'w'), indent=1, ensure_ascii=False)
print(open('../logs/10_my_photos.txt').read())
