const $=s=>document.querySelector(s);
const MAXS=2048, MAXL=40, MODEL='Xenova/segformer-b0-finetuned-ade-512-512';
const TF='https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2/+esm';
let src=null,ov=null,W=0,H=0,layers=[],seg=null,busy=false,manual=false,cnt=0,tok=0;
const cv=(w,h)=>Object.assign(document.createElement('canvas'),{width:w,height:h});
const status=(t,p)=>{$('#stat').textContent=t;if(p===null)$('#bar').removeAttribute('value');else if(p!==undefined)$('#bar').value=p};
const slug=s=>s.toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'')||'layer';
const pad=n=>String(n).padStart(2,'0');
const fname=(l,i)=>`layer_${pad(i+1)}_${slug(l.name)}.png`;
function save(blob,name){const a=document.createElement('a'),u=URL.createObjectURL(blob);a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),4000)}

function reset(){layers=[];cnt=0;src=ov=null;$('#stage').innerHTML='<p class="empty">Belum ada gambar.</p>';render()}

$('#file').onchange=async e=>{
  const f=e.target.files[0];if(!f||busy)return;reset();
  const url=URL.createObjectURL(f),img=new Image();
  try{await new Promise((ok,no)=>{img.onload=ok;img.onerror=no;img.src=url})}
  catch{status('File bukan gambar yang bisa dibuka. Pilih JPG, PNG, atau WebP.',0);URL.revokeObjectURL(url);return}
  const k=Math.min(1,MAXS/Math.max(img.naturalWidth,img.naturalHeight));
  W=Math.round(img.naturalWidth*k);H=Math.round(img.naturalHeight*k);
  src=cv(W,H);src.getContext('2d').drawImage(img,0,0,W,H);
  URL.revokeObjectURL(url);img.src='';
  ov=cv(W,H);ov.className='ov';
  const st=$('#stage');st.innerHTML='';st.append(src,ov);bindLasso();
  $('#info').textContent=`Ukuran kerja ${W}×${H}px`+(k<1?` (disusutkan dari ${Math.round(W/k)}×${Math.round(H/k)})`:' (ukuran asli)');
  $('#run').disabled=$('#manual').disabled=false;status('Siap. Tekan "Pisahkan dengan AI".',0);
};

async function getSeg(){
  if(seg)return seg;
  status('Memuat model AI…',0);
  const {pipeline,env}=await import(TF);env.allowLocalModels=false;
  seg=await pipeline('image-segmentation',MODEL,{dtype:'q8',progress_callback:p=>{if(p.status==='progress')status(`Mengunduh model ${Math.round(p.progress)}%`,p.progress)}});
  return seg;
}

function comps(d,w,h,minA){
  const lab=new Uint8Array(w*h),q=new Int32Array(w*h),res=[];
  for(let s=0;s<w*h;s++){
    if(d[s]<128||lab[s])continue;
    let a=0,b=1,x0=w,y0=h,x1=0,y1=0;q[0]=s;lab[s]=1;
    while(a<b){
      const i=q[a++],x=i%w,y=(i/w)|0;
      if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y;
      if(x>0&&d[i-1]>=128&&!lab[i-1]){lab[i-1]=1;q[b++]=i-1}
      if(x<w-1&&d[i+1]>=128&&!lab[i+1]){lab[i+1]=1;q[b++]=i+1}
      if(y>0&&d[i-w]>=128&&!lab[i-w]){lab[i-w]=1;q[b++]=i-w}
      if(y<h-1&&d[i+w]>=128&&!lab[i+w]){lab[i+w]=1;q[b++]=i+w}
    }
    if(b>=minA)res.push({n:b,m:q.slice(0,b),bb:[x0,y0,x1+1,y1+1]});
  }
  return res;
}

async function addLayer(name,paint,bb,post){
  const fc=cv(W,H),x=fc.getContext('2d');
  x.imageSmoothingQuality='high';paint(x);
  x.globalCompositeOperation='source-in';x.drawImage(src,0,0);
  if(post){x.globalCompositeOperation='source-over';post(x)}
  const blob=await new Promise(r=>fc.toBlob(r,'image/png'));
  const t=cv(96,Math.max(1,Math.round(96*H/W)));t.getContext('2d').drawImage(fc,0,0,t.width,t.height);
  const thumb=t.toDataURL();fc.width=fc.height=t.width=0;
  layers.push({name,blob,thumb,vis:true,bb});
}

async function runFoto(){
    const s=await getSeg();
    status('Menganalisis gambar…',null);
    const k=Math.min(1,768/Math.max(W,H)),sw=Math.round(W*k),sh=Math.round(H*k);
    const sc=cv(sw,sh);sc.getContext('2d').drawImage(src,0,0,sw,sh);
    const u=URL.createObjectURL(await new Promise(r=>sc.toBlob(r,'image/jpeg',.9)));sc.width=0;
    let out;try{out=await s(u)}finally{URL.revokeObjectURL(u)}
    const all=[];
    for(const o of out){
      const m=o.mask;
      for(const c of comps(m.data,m.width,m.height,Math.max(40,m.width*m.height*.004)))all.push({...c,label:o.label,w:m.width,h:m.height});
    }
    if(!all.length)throw new Error('Model tidak menemukan area yang cukup besar.');
    all.sort((a,b)=>b.n-a.n);const pick=all.slice(0,MAXL);
    const per={};
    for(let j=0;j<pick.length;j++){
      const c=pick[j];status(`Membuat layer ${j+1}/${pick.length}`,(j+1)/pick.length*100);
      const mc=cv(c.w,c.h),id=mc.getContext('2d').createImageData(c.w,c.h);
      for(const i of c.m)id.data[i*4+3]=255;
      mc.getContext('2d').putImageData(id,0,0);
      per[c.label]=(per[c.label]||0)+1;
      const kx=W/c.w,ky=H/c.h;
      await addLayer(`${c.label.split(',')[0]} ${per[c.label]}`,x=>x.drawImage(mc,0,0,W,H),[c.bb[0]*kx|0,c.bb[1]*ky|0,Math.ceil(c.bb[2]*kx),Math.ceil(c.bb[3]*ky)]);
      mc.width=0;
      if(j%4==3)render();
    }
    status(`Selesai: ${pick.length} layer. Hasil sesuai kemampuan model; sisihkan atau tambah layer secara manual bila perlu.`,100);
}

function dil(m,w,h,rx,ry){
  const t=new Uint8Array(w*h),o=new Uint8Array(w*h);
  for(let y=0;y<h;y++){let c=0;const b=y*w;for(let x=0;x<w+rx;x++){
    if(x<w)c+=m[b+x];if(x-2*rx-1>=0)c-=m[b+x-2*rx-1];
    const cx=x-rx;if(cx>=0&&cx<w&&c>0)t[b+cx]=1}}
  for(let x=0;x<w;x++){let c=0;for(let y=0;y<h+ry;y++){
    if(y<h)c+=t[y*w+x];if(y-2*ry-1>=0)c-=t[(y-2*ry-1)*w+x];
    const cy=y-ry;if(cy>=0&&cy<h&&c>0)o[cy*w+x]=255}}
  return o;
}
const mcv=(w,h,fn)=>{const c=cv(w,h),g=c.getContext('2d'),d=g.createImageData(w,h);fn(d.data);g.putImageData(d,0,0);return c};

async function runUI(){
  const k=Math.min(1,1600/Math.max(W,H)),w=Math.round(W*k),h=Math.round(H*k),N=w*h,kx=W/w,ky=H/h;
  status('Membaca warna…',null);
  const sc=cv(w,h),sg=sc.getContext('2d');sg.drawImage(src,0,0,w,h);
  const px=sg.getImageData(0,0,w,h).data;sc.width=0;
  const T=28,lab=new Int32Array(N).fill(-1),q=new Int32Array(N),cn=[],sd=[];
  for(let s=0;s<N;s++){
    if(lab[s]>=0)continue;
    const id=cn.length,r=px[s*4],g=px[s*4+1],b=px[s*4+2];let a=0,e=1;q[0]=s;lab[s]=id;
    const t=j=>{if(lab[j]>=0)return;const p=j*4;
      if(Math.abs(px[p]-r)<=T&&Math.abs(px[p+1]-g)<=T&&Math.abs(px[p+2]-b)<=T){lab[j]=id;q[e++]=j}};
    while(a<e){const i=q[a++],x=i%w;if(x>0)t(i-1);if(x<w-1)t(i+1);if(i>=w)t(i-w);if(i<N-w)t(i+w)}
    cn.push(e);sd.push(r,g,b);
  }
  status('Memisahkan elemen…',null);
  const big=cn.map((_,i)=>i).filter(i=>cn[i]>=N*.008).sort((a,b)=>cn[b]-cn[a]).slice(0,12);
  const isBig=new Uint8Array(cn.length);big.forEach(i=>isBig[i]=1);
  const det=new Uint8Array(N);for(let i=0;i<N;i++)if(!isBig[lab[i]])det[i]=1;
  const dm=dil(det,w,h,Math.max(2,Math.round(w*.012)),Math.max(1,Math.round(w*.004)));
  const groups=comps(dm,w,h,1).map(c=>{
    const m=c.m.filter(i=>det[i]);let x0=w,y0=h,x1=0,y1=0;
    for(const i of m){const x=i%w,y=(i/w)|0;if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y}
    return{m,bb:[x0,y0,x1+1,y1+1]};
  }).filter(c=>c.m.length>=6).sort((a,b)=>b.m.length-a.m.length).slice(0,70);
  const total=big.length+groups.length;let done=0;
  const sc2=bb=>[bb[0]*kx|0,bb[1]*ky|0,Math.ceil(bb[2]*kx),Math.ceil(bb[3]*ky)];
  let n=0;
  for(const id of big){
    status(`Membuat layer ${++done}/${total}`,done/total*100);
    const out=new Uint8Array(N);let a=0,e=0;
    const push=i=>{if(lab[i]!==id&&!out[i]){out[i]=1;q[e++]=i}};
    for(let x=0;x<w;x++){push(x);push((h-1)*w+x)}
    for(let y=0;y<h;y++){push(y*w);push(y*w+w-1)}
    while(a<e){const i=q[a++],x=i%w;if(x>0)push(i-1);if(x<w-1)push(i+1);if(i>=w)push(i-w);if(i<N-w)push(i+w)}
    let x0=w,y0=h,x1=0,y1=0;const R=sd[id*3],G=sd[id*3+1],B=sd[id*3+2];
    const mc=mcv(w,h,d=>{for(let i=0;i<N;i++)if(lab[i]===id||!out[i]){d[i*4+3]=255;
      const x=i%w,y=(i/w)|0;if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y}});
    const hc=mcv(w,h,d=>{for(let i=0;i<N;i++)if(lab[i]!==id&&!out[i]){d[i*4]=R;d[i*4+1]=G;d[i*4+2]=B;d[i*4+3]=255}});
    await addLayer(`latar ${++n}`,x=>x.drawImage(mc,0,0,W,H),sc2([x0,y0,x1+1,y1+1]),x=>x.drawImage(hc,0,0,W,H));
    mc.width=hc.width=0;
  }
  n=0;
  for(const c of groups){
    status(`Membuat layer ${++done}/${total}`,done/total*100);
    const mc=mcv(w,h,d=>{for(const i of c.m)d[i*4+3]=255});
    await addLayer(`elemen ${++n}`,x=>x.drawImage(mc,0,0,W,H),sc2(c.bb));mc.width=0;
    if(done%6===0)render();
  }
  status(`Selesai: ${total} layer (${big.length} latar, ${groups.length} elemen). Ganti nama, hapus, atau tambah layer manual bila ada yang kurang.`,100);
}

$('#run').onclick=async()=>{
  if(!src||busy)return;busy=true;$('#run').disabled=true;layers=[];cnt=0;
  try{await($('#mode').value==='ui'?runUI():runFoto())}
  catch(e){console.error(e);status('Gagal: '+(e.message||e)+'. Jika memakai mode Foto, periksa koneksi internet.',0)}
  busy=false;$('#run').disabled=false;render();
};

function amXML(){
  const base=($('#base').value.trim()||'/storage/emulated/0/Download/layerx_export').replace(/\/$/,'');
  const e=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
  const f6=v=>v.toFixed(6),T=5000,id0=Date.now();let m='',s='';
  layers.forEach((l,i)=>{
    if(!l.vis)return;const f=fname(l,i),u='file://'+base+'/layers/'+f;
    m+=`  <media uri="${e(u)}" filename="${e(f)}" title="${e(f)}" type="image/png" size="${l.blob.size}" orientation="0" infoUpdated="${id0}" width="${W}" height="${H}" />\r\n`;
    s+=`  <shape id="${id0+i}" label="${e(l.name)}" startTime="0" endTime="${T}" fillType="media" fillImage="${e(u)}" mediaFillMode="fill" s=".rect">\r\n    <transform>\r\n      <location value="${f6(W/2)},${f6(H/2)},0.000000" />\r\n    </transform>\r\n    <property name="size" type="vec2" value="${f6(W)},${f6(H)}" />\r\n  </shape>\r\n`;
  });
  return `<?xml version='1.0' encoding='UTF-8' ?>\r\n<scene title="LayerX Studio" width="${W}" height="${H}" exportWidth="${W}" exportHeight="${H}" precompose="dynamicResolution" bgcolor="#ffffffff" totalTime="${T}" fps="30" modifiedTime="${id0}" amver="1028425" ffver="106" am="com.alightcreative.motion/5.0.273.1028425" amplatform="android" retime="freeze" retimeAdaptFPS="false">\r\n${m}${s}</scene>`;
}

function bindLasso(){
  let pts=null;const g=ov.getContext('2d');
  const pos=e=>{const r=ov.getBoundingClientRect();return[(e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height]};
  ov.onpointerdown=e=>{if(!manual)return;pts=[pos(e)];ov.setPointerCapture(e.pointerId)};
  ov.onpointermove=e=>{
    if(!pts)return;pts.push(pos(e));g.clearRect(0,0,W,H);g.beginPath();
    pts.forEach((p,i)=>i?g.lineTo(...p):g.moveTo(...p));
    g.lineWidth=Math.max(2,W/200);g.strokeStyle='#b6f23a';g.stroke();
  };
  ov.onpointerup=async()=>{
    if(!pts)return;const p=pts;pts=null;g.clearRect(0,0,W,H);
    if(p.length<5)return;
    const xs=p.map(a=>a[0]),ys=p.map(a=>a[1]);
    await addLayer(`manual ${++cnt}`,x=>{x.beginPath();p.forEach((a,i)=>i?x.lineTo(...a):x.moveTo(...a));x.closePath();x.fillStyle='#000';x.fill()},
      [Math.max(0,Math.min(...xs))|0,Math.max(0,Math.min(...ys))|0,Math.min(W,Math.ceil(Math.max(...xs))),Math.min(H,Math.ceil(Math.max(...ys)))]);
    render();
  };
}
$('#manual').onclick=()=>{
  manual=!manual;$('#manual').textContent='Seleksi manual: '+(manual?'aktif':'mati');
  $('#manual').classList.toggle('on',manual);ov&&ov.classList.toggle('on',manual);
};

function render(){
  const list=$('#list');list.innerHTML='';
  if(!layers.length)list.innerHTML='<p class="empty">Belum ada layer.</p>';
  layers.forEach((l,i)=>{
    const d=document.createElement('div');d.className='layer'+(l.vis?'':' hid');
    d.innerHTML=`<img alt=""><div><input type="text"><div class="acts"><button data-a="v" class="${l.vis?'':'off'}">${l.vis?'Tampil':'Sembunyi'}</button><button data-a="u">Naik</button><button data-a="d">Turun</button><button data-a="s">PNG</button><button data-a="x">Hapus</button></div></div>`;
    d.querySelector('img').src=l.thumb;
    const inp=d.querySelector('input');inp.value=l.name;inp.onchange=()=>l.name=inp.value.trim()||l.name;
    d.querySelector('.acts').onclick=e=>{
      const a=e.target.dataset.a;if(!a)return;
      if(a==='v')l.vis=!l.vis;
      else if(a==='x')layers.splice(i,1);
      else if(a==='u'&&i<layers.length-1)[layers[i],layers[i+1]]=[layers[i+1],layers[i]];
      else if(a==='d'&&i>0)[layers[i],layers[i-1]]=[layers[i-1],layers[i]];
      else if(a==='s'){save(l.blob,fname(l,i));return}
      render();
    };
    list.append(d);
  });
  $('#zip').disabled=$('#mix').disabled=!layers.length;
  drawComp();
}

async function drawComp(){
  const c=$('#comp'),t=++tok;
  if(!src){c.width=c.height=0;return}
  c.width=Math.min(W,720);c.height=Math.round(c.width*H/W);
  const x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);
  for(const l of layers){
    if(!l.vis)continue;
    const bm=await createImageBitmap(l.blob);
    if(t===tok)x.drawImage(bm,0,0,c.width,c.height);bm.close();
    if(t!==tok)return;
  }
}
$('#chk').onchange=e=>$('#comp').classList.toggle('checker',e.target.checked);

async function mixBlob(){
  const c=cv(W,H),x=c.getContext('2d');
  for(const l of layers){if(!l.vis)continue;const bm=await createImageBitmap(l.blob);x.drawImage(bm,0,0);bm.close()}
  const b=await new Promise(r=>c.toBlob(r,'image/png'));c.width=0;return b;
}
$('#mix').onclick=async()=>{try{save(await mixBlob(),'composite.png')}catch(e){status('Gagal membuat gambar gabungan: '+e.message)}};
$('#zip').onclick=async()=>{
  if(typeof JSZip==='undefined'){status('JSZip belum termuat. Periksa koneksi internet.');return}
  try{
    status('Menyusun ZIP…',null);
    const z=new JSZip(),meta={app:'LayerX Studio',width:W,height:H,note:'Semua layer berukuran width×height; tumpuk di posisi 0,0. bbox = area objek.',layers:[]};
    layers.forEach((l,i)=>{
      const f=fname(l,i);z.file('layers/'+f,l.blob);
      meta.layers.push({name:l.name,order:i+1,visible:l.vis,file:'layers/'+f,x:0,y:0,width:W,height:H,bbox:{x:l.bb[0],y:l.bb[1],width:l.bb[2]-l.bb[0],height:l.bb[3]-l.bb[1]}});
    });
    z.file('alightmotion.xml',amXML());
    z.file('composite.png',await mixBlob());
    z.file('metadata.json',JSON.stringify(meta,null,2));
    save(await z.generateAsync({type:'blob',compression:'STORE'}),'layerx_export.zip');
    status('ZIP diunduh.',100);
  }catch(e){status('Gagal membuat ZIP: '+e.message,0)}
};
