import argparse, json, shutil
from pathlib import Path
from PIL import Image

def main():
    parser=argparse.ArgumentParser(description="Build a Hugging Face imagefolder dataset with metadata.jsonl.")
    parser.add_argument("--input",required=True,help="Input JSONL with image and text fields.")
    parser.add_argument("--output-dir",default="data",help="Output folder containing images/ and metadata.jsonl.")
    args=parser.parse_args()
    source=Path(args.input); out=Path(args.output_dir); image_dir=out/"images"
    image_dir.mkdir(parents=True,exist_ok=True)
    good=bad=0
    with source.open(encoding="utf-8") as fin, (out/"metadata.jsonl").open("w",encoding="utf-8") as fout:
        for line_no,line in enumerate(fin,1):
            try:
                row=json.loads(line); original=Path(row["image"])
                if not original.is_absolute(): original=source.parent/original
                caption=str(row["text"]).strip()
                if not caption: raise ValueError("empty caption")
                with Image.open(original) as im: im.verify()
                target=image_dir/f"{good:07d}{original.suffix.lower()}"
                shutil.copy2(original,target)
                fout.write(json.dumps({"file_name":f"images/{target.name}","text":caption},ensure_ascii=False)+"\n")
                good+=1
            except Exception as exc:
                bad+=1; print(f"Skipping line {line_no}: {exc}")
    print(f"Valid={good}, skipped={bad}, dataset folder={out}")
if __name__=="__main__": main()
