import stable_whisper, random, glob, os, traceback

LANGUAGE='sk'

random.seed(0)
model = stable_whisper.load_model('base')

for file in sorted(glob.glob('*.mp3')):
    txt = file + '.txt'
    try:
        text = open(txt, encoding='utf-8').read().strip()
        result = model.align(file, text, language=LANGUAGE)
        result.adjust_gaps()
        result.save_as_json(file + '.json')
        print('OK ', file)
    except Exception:
        print('FAIL', file)
        traceback.print_exc()

