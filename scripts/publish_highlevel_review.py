#!/usr/bin/env python3
"""Publish an already-reviewed RGB image and resulting high-level plan."""
import argparse,datetime,json,os,pathlib,shutil,tempfile

parser=argparse.ArgumentParser()
parser.add_argument('image')
parser.add_argument('--prompt',default='')
parser.add_argument('--visual',default='{}',help='JSON with route_pixels and optional goal_pixel')
args=parser.parse_args()
with open('/mnt/robotlogs/current-search.json') as handle:root=pathlib.Path(json.load(handle)['root'])
source=pathlib.Path(args.image).resolve()
if not source.is_file():raise ValueError('reviewed image does not exist')
visual=json.loads(args.visual)
if not isinstance(visual,dict):raise ValueError('visual must be a JSON object')
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
name='highlevel-reviewed-'+stamp.replace(':','-')+'.jpg'
shutil.copyfile(str(source),str(root/name))
record=dict(reviewed_at=stamp,review_image=name,prompt=args.prompt,high_level_visual=visual)
temporary=root/'highlevel-review.json.tmp'
temporary.write_text(json.dumps(record,indent=2))
os.replace(str(temporary),str(root/'highlevel-review.json'))
print(json.dumps(record))
