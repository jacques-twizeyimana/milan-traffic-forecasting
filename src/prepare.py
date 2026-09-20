"""Download, verify and aggregate one official daily file at a time."""
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
DOI = 'doi:10.7910/DVN/EGZHFV'
API = 'https://dataverse.harvard.edu/api'
COLS = ['square_id', 'timestamp', 'internet']


def aggregate(frame):
    """A missing activity field is not silently converted into zero."""
    return frame.groupby(COLS[:2], sort=False).internet.sum(min_count=1)


def rank_areas(totals):
    return totals.rename('total').rename_axis('square_id').reset_index().sort_values(
        ['total', 'square_id'], ascending=[False, True])


def sample_memory(path, optimized):
    options = dict(sep='\t', header=None, nrows=100_000)
    if optimized:
        options.update(usecols=[0, 1, 7], dtype={0:'int16', 1:'int64', 7:'float64'})
    df = pd.read_csv(path, **options)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(dataframe_bytes=int(df.memory_usage(deep=True).sum()),
                peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss * 1024),
                rows=len(df))


def download_response(session, file_id):
    url = f'{API}/access/datafile/{file_id}'
    # Official guestbook access; never invent an identity or commit contact details.
    email = os.environ.get('DATAVERSE_EMAIL')
    private = DATA / 'guestbook.json'
    if not email and private.exists():
        email = json.loads(private.read_text()).get('email')
    if email:
        response = session.post(url, json={'guestbookResponse': {'email': email}},
                                stream=True, timeout=(30,180))
    else:
        response = session.get(url, stream=True, timeout=(30,180))
    if response.status_code == 400 and 'Guestbook' in response.text:
        response.close()
        raise SystemExit('Official download requires an email guestbook response. Set DATAVERSE_EMAIL '
                         'to an email you authorize submitting to Harvard Dataverse, then rerun.')
    if 'application/json' in response.headers.get('Content-Type','') and response.ok:
        payload = response.json()
        response.close()
        signed_url = payload.get('data', {}).get('signedUrl')
        if not signed_url:
            raise RuntimeError('Official access response did not contain a signed download URL')
        return session.get(signed_url, stream=True, timeout=(30,180))
    return response


def main():
    start = time.perf_counter()
    for d in [DATA/'daily', DATA/'raw', OUT/'tables']:
        d.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    manifest_path = OUT/'dataset_manifest.json'
    if not manifest_path.exists():
        response = session.get(f'{API}/datasets/:persistentId/?persistentId={DOI}', timeout=90)
        response.raise_for_status()
        manifest_path.write_text(json.dumps(response.json(), indent=2))
    files = json.loads(manifest_path.read_text())['data']['latestVersion']['files']
    for number, entry in enumerate(files, 1):
        meta = entry['dataFile']
        name = meta['filename']
        dest = DATA/'daily'/name.replace('.txt', '.parquet')
        audit = dest.with_suffix('.json')
        if dest.exists() and audit.exists():
            print(f'cached {number}/{len(files)} {name}', flush=True)
            continue
        raw = DATA/'raw'/name
        print(f'downloading {number}/{len(files)} {name}', flush=True)
        for attempt in range(4):
            try:
                if raw.exists() and raw.stat().st_size == meta['filesize']:
                    with raw.open('rb') as stream: cached_digest = hashlib.file_digest(stream,'md5')
                    if cached_digest.hexdigest() == meta['md5']:
                        print('reusing verified raw file', flush=True)
                        break
                with download_response(session, meta['id']) as response:
                    response.raise_for_status()
                    url = response.url
                    supports_ranges = response.headers.get('Accept-Ranges') == 'bytes'
                    if not supports_ranges:
                        with raw.open('wb') as stream:
                            for chunk in response.iter_content(1024*1024): stream.write(chunk)
                if supports_ranges:
                    # Eight bounded transfers within ONE daily file; files remain sequential.
                    size = meta['filesize']
                    with raw.open('wb') as stream: stream.truncate(size)
                    def transfer(bounds):
                        low, high = bounds
                        with requests.get(url, headers={'Range':f'bytes={low}-{high}'},
                                          stream=True, timeout=(30,180)) as part:
                            part.raise_for_status()
                            assert part.status_code == 206, 'Server ignored byte range'
                            written = 0
                            with raw.open('r+b') as stream:
                                stream.seek(low)
                                for chunk in part.iter_content(1024*1024):
                                    stream.write(chunk); written += len(chunk)
                            assert written == high-low+1, 'Incomplete byte range'
                    step = (size+7)//8
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        list(pool.map(transfer, [(low,min(size-1,low+step-1)) for low in range(0,size,step)]))
                assert raw.stat().st_size == meta['filesize'], 'File size mismatch'
                with raw.open('rb') as stream: digest = hashlib.file_digest(stream,'md5')
                assert digest.hexdigest() == meta['md5'], 'Checksum mismatch'
                break
            except (requests.RequestException, AssertionError):
                if attempt == 3: raise
                time.sleep(2 ** attempt)
        if not (OUT/'memory.json').exists():
            measurements = {}
            for mode in ['baseline','optimized']:
                result = subprocess.check_output([sys.executable, __file__, '--memory', str(raw), mode], text=True)
                measurements[mode] = json.loads(result)
            (OUT/'memory.json').write_text(json.dumps(measurements, indent=2))
        parts, rows, missing = [], 0, 0
        for frame in pd.read_csv(raw, sep='\t', header=None, usecols=[0,1,7],
                                 names=COLS, dtype={'square_id':'int16','timestamp':'int64','internet':'float64'},
                                 chunksize=250_000):
            assert frame.square_id.between(1,10_000).all()
            assert (frame.timestamp % 600_000 == 0).all()
            assert (frame.internet.dropna() >= 0).all()
            rows += len(frame); missing += int(frame.internet.isna().sum())
            parts.append(aggregate(frame))
        result = pd.concat(parts).groupby(level=[0,1]).sum(min_count=1).reset_index()
        temp = dest.with_suffix('.tmp')
        result.to_parquet(temp, index=False, compression='zstd')
        check = pd.read_parquet(temp)
        assert len(check) == len(result) and not check.duplicated(COLS[:2]).any()
        temp.replace(dest)
        audit.write_text(json.dumps(dict(source=name, md5=meta['md5'], raw_rows=rows,
            blank_internet_rows=missing, aggregate_rows=len(result), raw_bytes=meta['filesize'],
            aggregate_bytes=dest.stat().st_size), indent=2))
        raw.unlink()
        print(f'aggregated {len(result):,} area/intervals', flush=True)
    totals = pd.Series(0., index=pd.Index(range(1,10001), name='square_id'))
    for path in sorted((DATA/'daily').glob('*.parquet')):
        df = pd.read_parquet(path)
        totals = totals.add(df.groupby('square_id').internet.sum(min_count=1), fill_value=0)
    ranking = rank_areas(totals)
    ranking.to_csv(OUT/'tables/area_totals.csv', index=False)
    ids = list(dict.fromkeys(ranking.square_id.head(3).tolist()+[4159,4556]))
    selected = []
    for path in sorted((DATA/'daily').glob('*.parquet')):
        df = pd.read_parquet(path, filters=[('square_id','in',ids)])
        selected.append(df)
    df = pd.concat(selected).groupby(COLS[:2]).internet.sum(min_count=1).reset_index()
    df['time'] = pd.to_datetime(df.timestamp, unit='ms', utc=True).dt.tz_convert('Europe/Rome')
    wide = df.pivot(index='time', columns='square_id', values='internet').sort_index()
    wide = wide.reindex(pd.date_range(wide.index.min(),wide.index.max(),freq='10min'))
    wide.to_parquet(DATA/'selected.parquet')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = dict(files=len(files), source_bytes=sum(f['dataFile']['filesize'] for f in files),
        aggregate_bytes=sum(p.stat().st_size for p in (DATA/'daily').glob('*.parquet')),
        top_three=ids[:3], start=str(wide.index.min()), end=str(wide.index.max()),
        missing_selected={str(k):int(v) for k,v in wide.isna().sum().items()},
        elapsed_seconds=time.perf_counter()-start,
        peak_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024))
    prior = OUT/'preparation.json'
    if prior.exists():
        # Preserve original download/preparation timing during cached verification runs.
        previous = json.loads(prior.read_text())
        summary['initial_elapsed_seconds'] = previous.get('initial_elapsed_seconds',previous['elapsed_seconds'])
        summary['initial_peak_rss_bytes'] = previous.get('initial_peak_rss_bytes',previous['peak_rss_bytes'])
    prior.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--memory':
        print(json.dumps(sample_memory(sys.argv[2],sys.argv[3]=='optimized')))
    else: main()
