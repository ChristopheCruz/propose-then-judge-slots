"use strict";
(function(){
const $=id=>document.getElementById(id),L=window.L,RAW=window.EXPLAIN_DATA;
const nf=(x,d=3)=>x.toLocaleString(L.locale,{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=(x,d=1)=>nf(100*x,d)+" %";
const v=n=>`var(--${n})`;
const fill=(t,o)=>t.replace(/\{(\w)\}/g,(m,k)=>o[k]);
const LISTS=[["reel",L.real],["oracle",L.oracle]];
function tabs(id,items,value,on){const el=$(id);
  const draw=()=>el.querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.v===value));
  items.forEach(([k,l])=>{const b=document.createElement("button");b.type="button";b.textContent=l;b.dataset.v=k;b.onclick=()=>{value=k;draw();on(k)};el.appendChild(b)});draw()}
const soft=(p,T)=>{const z=p.map(x=>Math.log(x/1e4)/T),m=Math.max(...z),e=z.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s)};
const cache=new Map();
function rows(list,T){const key=list+T;if(cache.has(key))return cache.get(key);
  const out=RAW[list].q.map(q=>{const t=q[0],p=soft(q.slice(1,-1),T),none=p.length-1;let k=0;p.forEach((x,i)=>{if(x>p[k])k=i});
    return [p[k],k===t,k===t?(k===none?1:0):(k===none?3:2)]});
  out.sort((a,b)=>b[0]-a[0]);if(cache.size>40)cache.clear();cache.set(key,out);return out}

let ex=0;
function drawEx(){const e=L.ex[ex],T=+$("exT").value,tau=+$("exTau").value;
  $("exTo").textContent=nf(T,2);$("exTauo").textContent=nf(tau,2);$("exU").textContent=e.u;$("exS").textContent=L.slot+e.slot;
  const z=e.o.map(o=>o[1]/T),m=Math.max(...z),x=z.map(a=>Math.exp(a-m)),s=x.reduce((a,b)=>a+b,0),p=x.map(a=>a/s),none=p.length-1;
  let k=0;p.forEach((a,i)=>{if(a>p[k])k=i});
  $("exO").innerHTML=e.o.map((o,i)=>`<div class="o"><span>${o[0]}</span><span class="br"><i class="${i===none?"n":""}" style="width:${100*p[i]}%"></i></span><b>${nf(p[i],2)}</b></div>`).join("");
  const P=$("exP");let txt,col,why;const a={o:e.o[k][0],p:nf(p[k],2),t:nf(tau,2)};
  if(p[k]<tau){txt=L.ask;col="ask";why=fill(L.why_ask,a)}else if(k===none){txt=L.ignore;col="ign";why=fill(L.why_ign,a)}else{txt=`${L.act}${L.sep}${e.slot} = ${e.o[k][0]}`;col="good";why=fill(L.why_act,a)}
  P.textContent=txt;P.style.background=v(col);
  $("exW").textContent=why+(p[k]>=tau&&k!==e.gold?fill(L.why_err,{g:e.o[e.gold][0]}):"")}
tabs("exTabs",L.ex.map((e,i)=>[String(i),e.name]),"0",k=>{ex=+k;drawEx()});
["exT","exTau"].forEach(i=>$(i).oninput=drawEx);

let cl="reel";
function drawCal(){const T=+$("calT").value;$("calTo").textContent=nf(T,2);
  const r=rows(cl,T),B=15,b=Array.from({length:B},()=>[0,0,0]);r.forEach(([c,ok])=>{const x=b[Math.min(Math.floor(c*B),B-1)];x[0]++;x[1]+=c;x[2]+=ok});
  const n=r.length,ece=b.reduce((a,x)=>a+Math.abs(x[1]-x[2]),0)/n,W=560,H=330,Lf=70,R=30,Tp=12,Bt=46,sx=a=>Lf+a*(W-Lf-R),sy=a=>H-Bt-a*(H-Tp-Bt);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${L.y_right}">`;
  [0,.25,.5,.75,1].forEach(a=>{s+=`<line class="gr" x1="${sx(0)}" x2="${sx(1)}" y1="${sy(a)}" y2="${sy(a)}"/><text x="${Lf-8}" y="${sy(a)+4}" text-anchor="end">${nf(a,2)}</text><text x="${sx(a)}" y="${H-Bt+18}" text-anchor="middle">${nf(a,2)}</text>`});
  s+=`<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="${v("mute")}" stroke-dasharray="5 4"/><text x="${(Lf+W-R)/2}" y="${H-8}" text-anchor="middle">${L.x_conf}</text><text transform="translate(13 ${(Tp+H-Bt)/2}) rotate(-90)" text-anchor="middle">${L.y_right}</text>`;
  b.forEach(x=>{if(!x[0])return;const c=x[1]/x[0],a=x[2]/x[0];s+=`<line x1="${sx(c)}" x2="${sx(c)}" y1="${sy(c)}" y2="${sy(a)}" stroke="${v("bad")}" stroke-width="1.5"/><circle cx="${sx(c)}" cy="${sy(a)}" r="${3+11*Math.sqrt(x[0]/n)}" fill="${v("accent")}" fill-opacity=".85"/>`});
  $("calC").innerHTML=s+"</svg>";$("calE").textContent=nf(ece);$("calA").textContent=pct(r.reduce((a,x)=>a+x[1],0)/n);$("calS").textContent=pct(r.filter(x=>x[0]>=.99).length/n)}
tabs("calL",LISTS,cl,k=>{cl=k;drawCal()});$("calT").oninput=drawCal;$("calR").onclick=()=>{$("calT").value=RAW.reel.T;drawCal()};

let dl="reel",best=.8;const K=[[L.k0,"good"],[L.k1,"ign"],[L.k2,"bad"],[L.k3,"mute"],[L.k4,"ask"]];
$("decG").innerHTML=K.map(k=>`<span><i style="background:${v(k[1])}"></i>${k[0]}</span>`).join("");
function ev(r,t,c){let tot=0;const n=[0,0,0,0,0];for(const [p,,g] of r){if(p>=t){n[g]++;tot+=g===0?1:g===1?0:g===2?-c.w:-c.m}else{n[4]++;tot-=c.a}}return{R:tot/r.length,n}}
function drawDec(){const c={w:+$("cW").value,a:+$("cA").value,m:+$("cM").value},tau=+$("tau").value;
  $("cWo").textContent=nf(c.w,1);$("cAo").textContent=nf(c.a,2);$("cMo").textContent=nf(c.m,2);$("tauo").textContent=nf(tau,3);
  const r=rows(dl,RAW[dl].T),g=[];for(let t=.3;t<=1.0001;t+=.005)g.push(t);const cv=g.map(t=>[t,ev(r,t,c).R]);
  let b=cv[0];cv.forEach(x=>{if(x[1]>b[1])b=x});best=b[0];
  const cur=ev(r,tau,c),lo=Math.min(...cv.map(x=>x[1])),hi=Math.max(...cv.map(x=>x[1])),pad=(hi-lo)*.12||.01,W=560,H=250,Lf=62,R=30,Tp=12,Bt=44,sx=a=>Lf+(a-.3)/.7*(W-Lf-R),sy=a=>H-Bt-(a-lo+pad)/(hi-lo+2*pad)*(H-Tp-Bt);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${L.r_cur}">`;
  for(let i=0;i<=4;i++){const a=lo-pad+(hi-lo+2*pad)*i/4;s+=`<line class="gr" x1="${Lf}" x2="${W-R}" y1="${sy(a)}" y2="${sy(a)}"/><text x="${Lf-8}" y="${sy(a)+4}" text-anchor="end">${nf(Math.abs(a)<.005?0:a,2)}</text>`}
  [.3,.5,.7,.9,1].forEach(a=>{s+=`<text x="${sx(a)}" y="${H-Bt+18}" text-anchor="middle">${nf(a,1)}</text>`});
  s+=`<text x="${(Lf+W-R)/2}" y="${H-6}" text-anchor="middle">${L.x_thr}</text><polyline fill="none" stroke="${v("accent")}" stroke-width="2.5" points="${cv.map(x=>sx(x[0]).toFixed(1)+","+sy(x[1]).toFixed(1)).join(" ")}"/><line x1="${sx(b[0])}" x2="${sx(b[0])}" y1="${Tp}" y2="${H-Bt}" stroke="${v("good")}" stroke-dasharray="4 4"/><line x1="${sx(tau)}" x2="${sx(tau)}" y1="${Tp}" y2="${H-Bt}" stroke="${v("ink")}"/><circle cx="${sx(tau)}" cy="${sy(cur.R)}" r="5" fill="${v("ink")}"/>`;
  $("decC").innerHTML=s+"</svg>";
  const n=r.length;let x=0,bar=`<svg viewBox="0 0 560 26" role="img" aria-label="${L.asked}">`;[0,1,2,3,4].forEach(i=>{const w=560*cur.n[i]/n;bar+=`<rect x="${x}" y="2" width="${w}" height="22" fill="${v(K[i][1])}"/>`;x+=w});
  $("decB").innerHTML=bar+"</svg>";$("dR").textContent=nf(cur.R);$("dB").textContent=nf(b[1]);$("dT").textContent=nf(b[0],3);$("dA").textContent=pct(cur.n[4]/n)}
tabs("decL",LISTS,dl,k=>{dl=k;drawDec()});["cW","cA","cM","tau"].forEach(i=>$(i).oninput=drawDec);$("dO").onclick=()=>{$("tau").value=best;drawDec()};

const RES=[["MTOP",.953,.488,.578],["Multi3WOZ",.912,.537,.536],["MASSIVE",.904,.524,.623],["MASSIVE *",.822,.466,.627],["SGD",.848,.515,.526]];
let rl="reel";
$("resG").innerHTML=`<span><i style="background:${v("accent")}"></i>F1</span><span id="resR"><i style="background:${v("ink")}"></i>${L.recall}</span><span>* sw, cy, is, km, am</span>`;
function drawRes(){$("resR").hidden=rl!=="reel";const W=560,rh=42,Lf=112,R=62,H=rh*RES.length+6,sx=a=>Lf+a*(W-Lf-R);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="F1">`;
  RES.forEach((d,i)=>{const y=6+i*rh,f=rl==="reel"?d[2]:d[1];s+=`<text x="${Lf-10}" y="${y+21}" text-anchor="end">${d[0]}</text><rect x="${Lf}" y="${y+4}" width="${W-Lf-R}" height="26" fill="${v("track")}"/><rect x="${Lf}" y="${y+4}" width="${sx(f)-Lf}" height="26" fill="${v("accent")}"/><text x="${W-R+8}" y="${y+21}" class="b">${nf(f)}</text>`;
    if(rl==="reel")s+=`<line x1="${sx(d[3])}" x2="${sx(d[3])}" y1="${y}" y2="${y+34}" stroke="${v("ink")}" stroke-width="3"/>`});
  $("resC").innerHTML=s+"</svg>"}
tabs("resL",LISTS,rl,k=>{rl=k;drawRes()});

const F={o:.902,r:.517},FP={o:.009,r:.110};
function drawMix(){const x=+$("mix").value/100,vol=Math.round(10**+$("vol").value);$("mixo").textContent=pct(x,0);$("volo").textContent=vol.toLocaleString(L.locale);
  const f=s=>(1-s)*F.o+s*F.r,W=560,H=230,Lf=48,R=34,Tp=12,Bt=44,lo=.5,hi=.92,sx=a=>Lf+a*(W-Lf-R),sy=a=>H-Bt-(a-lo)/(hi-lo)*(H-Tp-Bt);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${L.exp_f1}">`;
  [.5,.6,.7,.8,.9].forEach(a=>{s+=`<line class="gr" x1="${Lf}" x2="${W-R}" y1="${sy(a)}" y2="${sy(a)}"/><text x="${Lf-8}" y="${sy(a)+4}" text-anchor="end">${nf(a,1)}</text>`});
  [0,.25,.5,.75,1].forEach(a=>{s+=`<text x="${sx(a)}" y="${H-Bt+18}" text-anchor="middle">${nf(100*a,0)} %</text>`});
  s+=`<text x="${(Lf+W-R)/2}" y="${H-6}" text-anchor="middle">${L.x_mix}</text><line x1="${sx(0)}" y1="${sy(f(0))}" x2="${sx(1)}" y2="${sy(f(1))}" stroke="${v("accent")}" stroke-width="3"/><line x1="${sx(x)}" x2="${sx(x)}" y1="${Tp}" y2="${H-Bt}" stroke="${v("ink")}"/><circle cx="${sx(x)}" cy="${sy(f(x))}" r="5" fill="${v("ink")}"/>`;
  $("mixC").innerHTML=s+"</svg>";$("mF").textContent=nf(f(x));$("mW").textContent=Math.round(vol*((1-x)*FP.o+x*FP.r)).toLocaleString(L.locale)}
["mix","vol"].forEach(i=>$(i).oninput=drawMix);

drawEx();drawCal();drawDec();drawRes();drawMix();
const root=document.documentElement,btn=$("theme");let saved=null;try{saved=localStorage.getItem("theme")}catch(e){}
if(saved==="light"||saved==="dark")root.setAttribute("data-theme",saved);
btn.addEventListener("click",()=>{const dark=root.getAttribute("data-theme")==="dark"||(!root.getAttribute("data-theme")&&matchMedia("(prefers-color-scheme: dark)").matches),next=dark?"light":"dark";root.setAttribute("data-theme",next);try{localStorage.setItem("theme",next)}catch(e){}});
})();
