"""Create standalone HTML/PDF and plots, with CSV chart data beside them."""
import argparse,csv,json,re,html,textwrap,base64
from pathlib import Path
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,Preformatted,KeepTogether
from reportlab.lib.pagesizes import A4

STUDY=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--source', default='REPORT.md', help='Markdown path relative to the study folder')
parser.add_argument('--output-stem', default='REPORT', help='Output name relative to the study folder')
args=parser.parse_args()
fig=STUDY/'figures';fig.mkdir(exist_ok=True)
source=(STUDY/args.source).read_text()
output=STUDY/args.output_stem
output.parent.mkdir(parents=True,exist_ok=True)
body=markdown.markdown(source,extensions=['tables','fenced_code','footnotes'])
for path in fig.glob('*.png'):
 body=body.replace('src="figures/'+path.name+'"','src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"')
css='body{font-family:system-ui,"Noto Sans CJK SC",sans-serif;max-width:1050px;margin:50px auto;padding:0 25px;color:#202020;line-height:1.75}h1{font-size:30px}h2{margin-top:2em;font-size:22px}h3{font-size:18px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:7px;text-align:left}th{background:#f3f3f3}img{max-width:100%}pre{background:#f5f5f5;padding:14px;overflow:auto}a{color:#285d88}@media print{body{margin:0;font-size:10pt}h2,h3{break-after:avoid}tr,img{break-inside:avoid}}'
output.with_suffix('.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LoRA SFT 泛化与因果研究</title><style>'+css+'</style><body>'+body+'</body></html>')
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
styles={
 'body':ParagraphStyle('body',fontName='STSong-Light',fontSize=9.5,leading=15,spaceAfter=7,wordWrap='CJK'),
 'h1':ParagraphStyle('h1',fontName='STSong-Light',fontSize=21,leading=28,spaceAfter=16,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName='STSong-Light',fontSize=14,leading=20,spaceBefore=14,spaceAfter=8,keepWithNext=True,wordWrap='CJK'),
 'h3':ParagraphStyle('h3',fontName='STSong-Light',fontSize=11,leading=17,spaceBefore=8,spaceAfter=5,keepWithNext=True,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='STSong-Light',fontSize=7.5,leading=11,wordWrap='CJK'),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=6.5,leading=9),
}
def inline(s):
 s=s.replace('−','-').replace('·',' / ').replace('θ̈','theta_ddot')
 s=html.escape(s)
 s=s.replace('ⱼ','<sub>j</sub>').replace('₁','<sub>1</sub>')
 s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'`([^`]*)`',r'\1',s)
 s=re.sub(r'\[\^(\d+)\]',r'<super>\1</super>',s)
 s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:'<link href="'+m.group(2)+'" color="#285d88">'+m.group(1)+'</link>',s)
 return s
def table_inline(s):
 # Keep per-seed numeric values intact instead of splitting a digit across lines.
 if s.count('/')>=2 and re.fullmatch(r'[+\-0-9./ ]+',s):
  values=s.split('/');groups=[];current=[]
  for value in values:
   if current and len(' / '.join(current+[value]))>12:
    groups.append(' / '.join(current));current=[]
   current.append(value)
  if current:groups.append(' / '.join(current))
  return '<br/>'.join(inline(x) for x in groups)
 return inline(s)
flow=[];lines=source.splitlines();i=0;width=A4[0]-86
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('```'):
  code=[];i+=1
  while i<len(lines) and not lines[i].startswith('```'):code.extend(textwrap.wrap(lines[i],width=100,replace_whitespace=False,drop_whitespace=False) or ['']);i+=1
  flow.append(Preformatted('\n'.join(code),styles['code']));flow.append(Spacer(1,9));i+=1;continue
 if line.startswith('|'):
  data=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   cells=[s.strip() for s in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch(r'[:\- ]+',s) for s in cells):data.append([Paragraph(table_inline(s),styles['cell']) for s in cells])
   i+=1
  n=len(data[0]);weights=[2]+[1]*(n-1);total=sum(weights)
  table=Table(data,colWidths=[width*w/total for w in weights],repeatRows=1,hAlign='LEFT')
  table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cccccc')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));flow.extend([table,Spacer(1,9)]);continue
 m=re.match(r'!\[[^\]]*\]\(([^)]+)\)',line)
 if m:
  path=STUDY/m.group(1);image=Image(str(path));image.drawHeight*=width/image.drawWidth;image.drawWidth=width;flow.extend([image,Spacer(1,9)]);i+=1;continue
 if line.startswith('#'):
  n=len(line)-len(line.lstrip('#'));flow.append(Paragraph(inline(line[n:].strip()),styles['h'+str(min(n,3))]));i+=1;continue
 footnote=re.match(r'\[\^(\d+)\]:\s*(.*)',line)
 if footnote:
  flow.append(Paragraph(footnote.group(1)+'. '+inline(footnote.group(2)),styles['body']));i+=1;continue
 list_item=re.match(r'^(?:[*+-]\s+|\d+\.\s+)',line)
 if list_item:
  flow.append(Paragraph(inline(line),styles['body']));i+=1;continue
 paragraph=[line];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','![','[^')) and not re.match(r'^(?:[*+-]\s+|\d+\.\s+)',lines[i].strip()):
  paragraph.append(lines[i].strip());i+=1
 flow.append(Paragraph(inline(' '.join(paragraph)),styles['body']))
def page_footer(canvas,doc):
 canvas.saveState();canvas.setFont('STSong-Light',8);canvas.setFillColor(colors.HexColor('#666666'))
 canvas.drawString(43,23,'LoRA SFT 泛化研究 / 2026-09-09')
 canvas.drawRightString(A4[0]-43,23,str(doc.page));canvas.restoreState()
SimpleDocTemplate(str(output.with_suffix('.pdf')),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=40,bottomMargin=40,title='LoRA SFT 泛化与因果研究',author='').build(flow,onFirstPage=page_footer,onLaterPages=page_footer)
print(f'Rendered {output.with_suffix(".html")} and {output.with_suffix(".pdf")}')
