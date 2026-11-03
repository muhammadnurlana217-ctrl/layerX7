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

async function addLayer(name,paint,bb){
  const fc=cv(W,H),x=fc.getContext('2d');
  x.imageSmoothingQuality='high';paint(x);
  x.globalCompositeOperation='source-in';x.drawImage(src,0,0);
  const blob=await new Promise(r=>fc.toBlob(r,'image/png'));
  const t=cv(96,Math.max(1,Math.round(96*H/W)));t.getContext('2d').drawImage(fc,0,0,t.width,t.height);
  const thumb=t.toDataURL();fc.width=fc.height=t.width=0;
  layers.push({name,blob,thumb,vis:true,bb});
}

$('#run').onclick=async()=>{
  if(!src||busy)return;busy=true;$('#run').disabled=true;
  try{
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
  }catch(e){
    console.error(e);
    status('Gagal: '+(e.message||e)+'. Periksa koneksi internet lalu coba lagi.',0);
  }
  busy=false;$('#run').disabled=false;render();
};

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
    z.file('composite.png',await mixBlob());
    z.file('metadata.json',JSON.stringify(meta,null,2));
    save(await z.generateAsync({type:'blob',compression:'STORE'}),'layerx_export.zip');
    status('ZIP diunduh.',100);
  }catch(e){status('Gagal membuat ZIP: '+e.message,0)}
};
