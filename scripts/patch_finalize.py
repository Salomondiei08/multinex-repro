"""Patch ~/multinex_repro/scripts/finalize.py: step 7 uses DICM/LIME/MEF/NPE instead of own photos."""
import os
p = os.path.expanduser("~/multinex_repro/scripts/finalize.py")
s = open(p).read()
a = s.index("# own photos (no-reference)"); b = s.index("res = dict(")
new = r'''# no-reference benchmarks (DICM, LIME, MEF, NPE) replace own photos at user request
def summ(log):
    t = rd(f'{L}/{log}.log')
    return {m[0]: dict(niqe=float(m[1]), brisque=float(m[2])) for m in
            re.findall(r'=== Dataset: (\w+) ===.*?NIQE:\s+([\d.]+)\nBRISQUE:\s+([\d.]+)', t, re.S)}
noref = dict(multinex=summ('07b_noref_multinex'), nano=summ('07c_noref_nano'))
for k in noref:
    v = list(noref[k].values())
    assert len(v) == 4, noref[k]
    noref[k]['mean_of_4'] = dict(niqe=float(np.mean([x['niqe'] for x in v])), brisque=float(np.mean([x['brisque'] for x in v])))
per = lambda rdir, ds: list(csv.DictReader(open(f'{M}/results/{rdir}/{ds}/per_image.csv')))
noref_examples = {}
for ds in ['DICM', 'LIME', 'MEF', 'NPE']:
    pmx = per('Multinex_NoRef', ds); pnn = {r['image']: r for r in per('Multinex-Nano_NoRef', ds)}
    ex = min(pmx, key=lambda r: float(r['niqe']))  # best Multinex NIQE in this dataset
    img = ex['image']; rel = os.path.splitext(img)[0] + '.png'
    cands = [q for q in glob.glob(f'{M}/data/{ds}/**/*', recursive=True)
             if os.path.relpath(q, f'{M}/data/{ds}') == img or os.path.basename(q) == os.path.basename(img)]
    noref_examples[ds] = dict(file=img, multinex_niqe=float(ex['niqe']), multinex_brisque=float(ex['brisque']),
                              nano_niqe=float(pnn[img]['niqe']), nano_brisque=float(pnn[img]['brisque']))
    side_by_side([(cands[0] if cands else None, f'{ds} input'),
                  (f'{M}/results/Multinex_NoRef/{ds}/images/{rel}', f"Multinex NIQE {float(ex['niqe']):.2f}"),
                  (f'{M}/results/Multinex-Nano_NoRef/{ds}/images/{rel}', f"Nano NIQE {float(pnn[img]['niqe']):.2f}"),
                  (None, 'No ground truth')], f'{fig}/noref_{ds}_{os.path.splitext(os.path.basename(img))[0]}.png')
    for rdir in ['Multinex_NoRef', 'Multinex-Nano_NoRef']:
        shutil.copy(f'{M}/results/{rdir}/{ds}/per_image.csv', f'{D}/noref_per_image_{rdir}_{ds}.csv')

'''
s = s[:a] + new + s[b:]
s = s.replace("mine=mine or None,",
              "mine=None, noref=noref, noref_examples=noref_examples,\n"
              "           paper_noref_table_note='Repo assets/NoRef_table.png shows pre-dagger numbers; README says the dagger checkpoints match the updated arXiv table, which is not in the repo. Not directly comparable.',")
s = s.replace("'Own photos: README no-reference script with the README NoRef weights (*_dagger_NoRef.pth, LOL-v2-synthetic option files).'",
              "'Own photos not provided; replaced at user request by the no-reference benchmarks DICM (69), LIME (10), MEF (17), NPE (8) from the BIMEF Google Drive folder, run with the unchanged README no-reference commands (*_dagger_NoRef.pth weights).'")
s = s.replace("['gpu', 'torch', 'lolv1', 'best_image', 'worst_image', 'mine']", "['best_image', 'worst_image', 'noref', 'noref_examples']")
assert "noref=noref" in s and "'noref_examples']" in s
open(p, "w").write(s)
print("patched")
