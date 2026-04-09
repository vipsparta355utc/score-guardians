// ═══════════════════════════════════════════════════════════════
// Score Guardians · Liga MX — Lógica principal
// Universidad Tres Culturas · Actualizado Marzo 2026
// Datos: cargados dinámicamente desde stats_equipos.json
// ═══════════════════════════════════════════════════════════════

'use strict';

// ──────────────────────────────────────────────────────────────
// SECCIÓN 1 · DATOS
// Cargados desde stats_equipos.json — NO editar manualmente
// Para actualizar: correr notebook y reemplazar el JSON
// ──────────────────────────────────────────────────────────────

const LOGOS={
  'America':'./imagenes/america.png','Chivas':'./imagenes/guadalajara.png',
  'Cruz Azul':'./imagenes/cruzazul.png','Pumas':'./imagenes/pumas.png',
  'Tigres':'./imagenes/tigres.png','Monterrey':'./imagenes/monterrey.png',
  'Santos':'./imagenes/santoslaguna.png','Leon':'./imagenes/leon.png',
  'Toluca':'./imagenes/toluca.png','Atlas':'./imagenes/atlas.png',
  'Pachuca':'./imagenes/pachuca.png','Tijuana':'./imagenes/tijuana.png',
  'Necaxa':'./imagenes/necaxa.png','Puebla':'./imagenes/puebla.png',
  'Queretaro':'./imagenes/gallosblancos.png','Mazatlan':'./imagenes/mazatlanfc.png',
  'Juarez':'./imagenes/cfjuarez.png','San Luis':'./imagenes/atlanticodesanluis.png'
};

// Variables globales — se llenan al cargar el JSON
let STATS = {};
let H2H_HISTORIAL = {};
let PROXIMOS = [];
let EQUIPOS_ORDENADOS = [];
let PROM_GOLES = 1.2112;
let FAC_LOCAL  = 1.15;
let MAX_GOLES  = 7;

// ── Carga del JSON ──────────────────────────────────────────────
// ──────────────────────────────────────────────────────────────
// MODELO RED NEURONAL — TensorFlow.js
// Pesos: PESO_NN=0.60 · PESO_POISSON=0.40 (igual que notebook)
// ──────────────────────────────────────────────────────────────
let NN_MODEL    = null;   // modelo tfjs cargado
let NN_SCALER   = null;   // {mean, scale, features}
let NN_LISTO    = false;  // flag: modelo disponible
const PESO_NN      = 0.60;
const PESO_POISSON = 0.40;

async function cargarModelo() {
  try {
    // Cargar scaler params
    const sr = await fetch('./tfjs_model/scaler_params.json');
    NN_SCALER = await sr.json();

    // Cargar modelo TF.js
    NN_MODEL = await tf.loadLayersModel('./tfjs_model/model.json');
    NN_LISTO = true;
    console.log('✅ Red neuronal cargada');

    // Actualizar badge en UI
    const badge = document.getElementById('nn-badge');
    if (badge) {
      badge.textContent = '🧠 NN Activa';
      badge.style.background = 'rgba(132,204,22,0.2)';
      badge.style.color = '#4d7c0f';
    }
  } catch(e) {
    console.warn('⚠️ Modelo NN no disponible, usando solo Poisson:', e.message);
    NN_LISTO = false;
  }
}

// ── Normalizar features con el scaler exportado ──────────────
function scalerTransform(featArr) {
  return featArr.map((v, i) => (v - NN_SCALER.mean[i]) / NN_SCALER.scale[i]);
}

// ── Predicción con red neuronal ───────────────────────────────
async function predecirNN(local, visita) {
  const sl = STATS[local], sv = STATS[visita];
  if (!sl || !sv) return null;

  const rawFeat = [
    sl.ppg,  sv.ppg,
    sl.gfpj, sv.gfpj,
    sl.gcpj, sv.gcpj,
    sl.pos_media || 9, sv.pos_media || 9,   // fallback si no hay pos
    sl.ppg  - sv.ppg,
    (sv.pos_media || 9) - (sl.pos_media || 9),
    sl.gfpj - sv.gfpj,
    sl.gcpj - sv.gcpj,
  ];

  const scaled   = scalerTransform(rawFeat);
  const tensor   = tf.tensor2d([scaled]);
  const pred     = NN_MODEL.predict(tensor);
  const probs    = await pred.data();
  tensor.dispose();
  pred.dispose();

  return { pL: probs[0], pE: probs[1], pV: probs[2] };
}

async function cargarDatos() {
  try {
    const res  = await fetch('./stats_equipos.json');
    const data = await res.json();

    // Config
    PROM_GOLES = data.config.promedio_goles_liga;
    FAC_LOCAL  = data.config.factor_localia;
    MAX_GOLES  = data.config.max_goles;

    // Stats — renombrar fa/fd a nombres internos
    STATS = {};
    for (const [eq, s] of Object.entries(data.stats)) {
      STATS[eq] = { ppg: s.ppg, gfpj: s.gfpj, gcpj: s.gcpj, fa: s.fa, fd: s.fd, ir: s.ir };
    }

    // H2H — convertir al formato que usa historial()
    H2H_HISTORIAL = {};
    for (const [key, partidos] of Object.entries(data.h2h)) {
      H2H_HISTORIAL[key] = partidos.map(p => ({ fecha: p.fecha, score: p.score }));
    }

    // Próximos partidos
    PROXIMOS = data.proximos;

    // Equipos ordenados por IR
    EQUIPOS_ORDENADOS = Object.keys(STATS).sort((a,b) => STATS[b].ir - STATS[a].ir);

    // Actualizar badge de jornada en el DOM
    const tag = document.querySelector('#sec-proximos .sec-tag');
    if (tag && data.config.jornada_actual) {
      tag.textContent = `Módulo 02 · Jornada ${data.config.jornada_actual} — CL 2026`;
    }

    // Iniciar la app
    initCharts();
    buildRanking();
    // Cargar modelo NN en paralelo (no bloquea la UI)
    cargarModelo().then(() => buildMatchCards());

  } catch(err) {
    console.error('Error cargando stats_equipos.json:', err);
    alert('No se pudo cargar stats_equipos.json. Verifica que el archivo esté en la misma carpeta.');
  }
}

// ──────────────────────────────────────────────────────────────
// SECCIÓN 2 · MODELO POISSON BASE
// poissonPMF · matrizPoisson · predecir
// ──────────────────────────────────────────────────────────────
function poissonPMF(k,lambda){
  if(lambda<=0)return k===0?1:0;
  let lp=-lambda+k*Math.log(lambda),lf=0;
  for(let i=2;i<=k;i++)lf+=Math.log(i);
  return Math.exp(lp-lf);
}

function matrizPoisson(ll,lv){
  const M=[];
  for(let i=0;i<=MAX_GOLES;i++){
    M[i]=[];
    for(let j=0;j<=MAX_GOLES;j++)M[i][j]=poissonPMF(i,ll)*poissonPMF(j,lv);
  }
  return M;
}

// ── Predicción principal ────────────────────────────────────────
function predecirPoisson(local,visita){
  const sl=STATS[local],sv=STATS[visita];
  if(!sl||!sv)return null;
  const ll=sl.fa*sv.fd*PROM_GOLES*FAC_LOCAL,lv=sv.fa*sl.fd*PROM_GOLES,M=matrizPoisson(ll,lv);
  let pL=0,pE=0,pV=0;
  for(let i=0;i<=MAX_GOLES;i++)for(let j=0;j<=MAX_GOLES;j++){if(i>j)pL+=M[i][j];else if(i===j)pE+=M[i][j];else pV+=M[i][j];}
  const tot=pL+pE+pV;pL/=tot;pE/=tot;pV/=tot;
  let maxP=0,gL=1,gV=0;
  for(let i=0;i<=MAX_GOLES;i++)for(let j=0;j<=MAX_GOLES;j++)if(M[i][j]>maxP){maxP=M[i][j];gL=i;gV=j;}
  const probs=[pL,pE,pV].sort((a,b)=>b-a),confianza=(probs[0]-probs[1])/probs[0]*100;
  return{ll:+ll.toFixed(3),lv:+lv.toFixed(3),pL:+(pL*100).toFixed(1),pE:+(pE*100).toFixed(1),pV:+(pV*100).toFixed(1),marcador:`${gL} - ${gV}`,confianza:+confianza.toFixed(1),M,sl,sv};
}
// ── Predicción ensemble: NN + Poisson ────────────────────────
async function predecir(local, visita) {
  const sl = STATS[local], sv = STATS[visita];
  if (!sl || !sv) return null;

  // Poisson base (síncrono)
  const rP = predecirPoisson(local, visita);
  if (!rP) return null;

  let pL, pE, pV;

  if (NN_LISTO) {
    // Ensemble: 60% NN + 40% Poisson
    const rNN = await predecirNN(local, visita);
    pL = PESO_NN * rNN.pL + PESO_POISSON * (rP.pL / 100);
    pE = PESO_NN * rNN.pE + PESO_POISSON * (rP.pE / 100);
    pV = PESO_NN * rNN.pV + PESO_POISSON * (rP.pV / 100);
    const tot = pL + pE + pV;
    pL /= tot; pE /= tot; pV /= tot;
    pL = +(pL * 100).toFixed(1);
    pE = +(pE * 100).toFixed(1);
    pV = +(pV * 100).toFixed(1);
  } else {
    // Solo Poisson si NN no está disponible
    pL = rP.pL; pE = rP.pE; pV = rP.pV;
  }

  const probs   = [pL, pE, pV].sort((a, b) => b - a);
  const confianza = +((probs[0] - probs[1]) / probs[0] * 100).toFixed(1);

  return {
    ll: rP.ll, lv: rP.lv,
    pL, pE, pV,
    marcador: rP.marcador,
    confianza,
    modoEnsemble: NN_LISTO,
    M: rP.M, sl, sv
  };
}


// ── Historial H2H real (últimos 5 enfrentamientos por par) ──────


// ──────────────────────────────────────────────────────────────
// SECCIÓN 3 · DATOS H2H
// H2H_HISTORIAL · historial
// ──────────────────────────────────────────────────────────────



// ──────────────────────────────────────────────────────────────
// SECCIÓN 4 · GRÁFICAS — Chart.js
// chartPred · destroyChart · drawPredChart · drawPoissonChart · initCharts
// ──────────────────────────────────────────────────────────────
function historial(local,visita){
  // Buscar en diccionario real
  const key=local+'_'+visita;
  if(H2H_HISTORIAL[key]) return H2H_HISTORIAL[key];
  // Buscar inverso (mismo partido, perspectiva opuesta)
  const keyInv=visita+'_'+local;
  if(H2H_HISTORIAL[keyInv]) return H2H_HISTORIAL[keyInv].map(p=>{
    const parts=p.score.split(' - ');
    return{fecha:p.fecha,score:`${parts[1]} - ${parts[0]}`};
  });
  // Fallback: generar con seed si no hay datos
  const fechas=['12/01/2025','20/10/2024','15/07/2024','18/03/2024','05/11/2023'],
        sl=STATS[local],sv=STATS[visita],
        seed=Array.from(local+visita).reduce((s,c)=>s+c.charCodeAt(0),0);
  return fechas.map((f,i)=>{
    const r=((seed*(i+3)*7)%100)/100,
          g1=Math.round(sl.gfpj*(0.5+r*0.8)),
          g2=Math.round(sv.gfpj*(0.5+((seed*(i+5))%100)/100*0.8));
    return{fecha:f,score:`${g1} - ${g2}`};
  });
}

let chartPred=null,chartPoisson=null,modalChartDona=null,modalChartLambda=null;
function destroyChart(c){if(c){try{c.destroy();}catch(e){}}return null;}

// Colores de paleta para gráficas
const C={
  blue:'rgba(26,58,143,0.75)',blueS:'rgba(26,58,143,1)',blueL:'rgba(74,124,255,0.75)',blueLS:'rgba(74,124,255,1)',
  lime:'rgba(132,204,22,0.75)',limeS:'rgba(132,204,22,1)',
  amber:'rgba(217,119,6,0.75)',amberS:'rgba(217,119,6,1)',
  grid:'rgba(26,58,143,0.08)',tick:'rgba(26,58,143,0.5)'
};

function drawPredChart(pL,pE,pV,local,visita){
  chartPred=destroyChart(chartPred);
  chartPred=new Chart(document.getElementById('chartPred').getContext('2d'),{
    type:'doughnut',
    data:{labels:[`Victoria ${local}`,'Empate',`Victoria ${visita}`],datasets:[{data:[pL,pE,pV],backgroundColor:[C.blue,C.amber,C.lime],borderColor:['#fff','#fff','#fff'],borderWidth:3}]},
    options:{responsive:true,plugins:{legend:{position:'bottom',labels:{color:C.tick,font:{size:11}}},tooltip:{callbacks:{label:c=>`${c.label}: ${c.parsed}%`}}},cutout:'62%'}
  });
}

function drawPoissonChart(ll,lv,local,visita){
  chartPoisson=destroyChart(chartPoisson);
  const labels=[...Array(8).keys()],dataL=labels.map(k=>+(poissonPMF(k,ll)*100).toFixed(2)),dataV=labels.map(k=>+(poissonPMF(k,lv)*100).toFixed(2));
  chartPoisson=new Chart(document.getElementById('chartPoisson').getContext('2d'),{
    type:'bar',
    data:{labels:labels.map(k=>`${k} gol${k!==1?'es':''}`),datasets:[{label:local,data:dataL,backgroundColor:C.blue,borderRadius:4},{label:visita,data:dataV,backgroundColor:C.lime,borderRadius:4}]},
    options:{responsive:true,plugins:{legend:{position:'top',labels:{color:C.tick,font:{size:11}}}},scales:{y:{title:{display:true,text:'Probabilidad (%)',color:C.tick},beginAtZero:true,ticks:{color:C.tick},grid:{color:C.grid}},x:{title:{display:true,text:'Goles anotados',color:C.tick},ticks:{color:C.tick},grid:{display:false}}}}
  });
  document.getElementById('poissonHint').style.display='none';
}

function initCharts(){
  const eq=EQUIPOS_ORDENADOS;
  const col=eq.map((_,i)=>`hsl(${220-i*8},${70-i*1.5}%,${45+i*1.5}%)`);

  new Chart(document.getElementById('chartRendimiento').getContext('2d'),{
    type:'bar',data:{labels:eq,datasets:[{label:'IR',data:eq.map(e=>STATS[e].ir),backgroundColor:col,borderRadius:4}]},
    options:{indexAxis:'y',responsive:true,plugins:{legend:{display:false}},scales:{x:{beginAtZero:true,max:100,ticks:{color:C.tick},grid:{color:C.grid}},y:{ticks:{color:C.tick,font:{size:10}},grid:{display:false}}}}
  });

  new Chart(document.getElementById('chartGoles').getContext('2d'),{
    type:'bar',
    data:{labels:eq,datasets:[{label:'GF/PJ',data:eq.map(e=>STATS[e].gfpj),backgroundColor:C.blue,borderRadius:3},{label:'GC/PJ',data:eq.map(e=>STATS[e].gcpj),backgroundColor:C.lime,borderRadius:3}]},
    options:{responsive:true,plugins:{legend:{position:'top',labels:{color:C.tick}}},scales:{x:{ticks:{maxRotation:45,font:{size:9},color:C.tick},grid:{display:false}},y:{beginAtZero:true,ticks:{color:C.tick},grid:{color:C.grid}}}}
  });

  new Chart(document.getElementById('chartPPG').getContext('2d'),{
    type:'line',
    data:{labels:eq,datasets:[{label:'PPG',data:eq.map(e=>STATS[e].ppg),borderColor:C.blueLS,backgroundColor:'rgba(74,124,255,0.08)',borderWidth:2.5,pointBackgroundColor:C.blueLS,pointRadius:3,fill:true,tension:0.3}]},
    options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{ticks:{maxRotation:45,font:{size:9},color:C.tick},grid:{display:false}},y:{beginAtZero:true,ticks:{color:C.tick},grid:{color:C.grid}}}}
  });

  const top8=EQUIPOS_ORDENADOS.slice(0,8);
  new Chart(document.getElementById('chartRadar').getContext('2d'),{
    type:'radar',
    data:{labels:top8,datasets:[
      {label:'Fuerza Ataque',data:top8.map(e=>STATS[e].fa),borderColor:C.blueLS,backgroundColor:'rgba(26,58,143,0.1)',borderWidth:2,pointBackgroundColor:C.blueLS},
      {label:'F.Defensa (inv.)',data:top8.map(e=>+(2-STATS[e].fd).toFixed(3)),borderColor:C.limeS,backgroundColor:'rgba(132,204,22,0.1)',borderWidth:2,pointBackgroundColor:C.limeS}
    ]},
    options:{responsive:true,scales:{r:{beginAtZero:true,min:0,max:1.5,ticks:{font:{size:9},color:C.tick,backdropColor:'transparent'},grid:{color:'rgba(26,58,143,0.12)'},pointLabels:{font:{size:10},color:C.tick},angleLines:{color:'rgba(26,58,143,0.1)'}}},plugins:{legend:{position:'top',labels:{color:C.tick,font:{size:10}}}}}
  });
}


// ──────────────────────────────────────────────────────────────
// SECCIÓN 5 · INTERFAZ
// buildRanking · buildMatchCards · predictMatch · openModal · closeModal
// updateTeamLogo · openFormulas · closeFormulas
// ──────────────────────────────────────────────────────────────
function buildRanking(){
  const tbody=document.getElementById('rankingBody'),maxIR=Math.max(...EQUIPOS_ORDENADOS.map(e=>STATS[e].ir));
  EQUIPOS_ORDENADOS.forEach((eq,i)=>{
    const s=STATS[eq],irPct=(s.ir/maxIR*100).toFixed(0),faC=s.fa>=1?'#1a3a8f':'#ef4444',fdC=s.fd<=1?'#4d7c0f':'#ef4444',rc=i<3?`r${i+1}`:'';
    tbody.innerHTML+=`<tr><td><span class="rank-g ${rc}">${i+1}</span></td><td>${eq}</td><td><strong>${s.ppg.toFixed(2)}</strong></td><td style="color:#1a3a8f;font-weight:700">${s.gfpj.toFixed(2)}</td><td style="color:#84cc16;font-weight:700">${s.gcpj.toFixed(2)}</td><td style="color:${faC};font-weight:700">${s.fa.toFixed(3)}</td><td style="color:${fdC};font-weight:700">${s.fd.toFixed(3)}</td><td><div class="ir-g"><strong style="color:#2251cc;min-width:32px;font-weight:800">${s.ir.toFixed(1)}</strong><div class="ir-g-bar"><div class="ir-g-fill" style="width:${irPct}%"></div></div></div></td></tr>`;
  });
}

async function buildMatchCards(){
  const grid=document.getElementById('matchesGrid');
  for(const m of PROXIMOS){
    const r=await predecir(m.local,m.visita);if(!r)continue;
    const modoTag = r.modoEnsemble ? '🧠 Ensemble' : '📊 Poisson';
    grid.innerHTML+=`<div class="match-g"><div class="match-date-g">${m.fecha}</div><div class="match-teams-g"><div class="team-g"><div class="team-logo-g"><img src="${LOGOS[m.local]||''}" alt="${m.local}" onerror="this.style.display='none'"></div><div class="team-name-g">${m.local}</div></div><div class="vs-g">VS</div><div class="team-g"><div class="team-logo-g"><img src="${LOGOS[m.visita]||''}" alt="${m.visita}" onerror="this.style.display='none'"></div><div class="team-name-g">${m.visita}</div></div></div><div class="pred-g"><h4>Predicción del Modelo <small style="font-size:0.75em;opacity:0.7">${modoTag}</small></h4><div class="pred-g-score">${r.marcador}</div><div class="pred-g-conf">Confianza: ${r.confianza}%</div><div class="pred-g-probs">L: ${r.pL}% · E: ${r.pE}% · V: ${r.pV}%</div></div><button class="detail-btn-g" onclick="openModal('${m.local}','${m.visita}')">Ver análisis detallado</button></div>`;
  }
}

async function predictMatch(){
  const local=document.getElementById('team1').value,visita=document.getElementById('team2').value;
  if(!local||!visita){alert('Selecciona ambos equipos');return;}
  if(local===visita){alert('Selecciona equipos diferentes');return;}
  const r=await predecir(local,visita);if(!r)return;

  document.getElementById('predictedScore').textContent=r.marcador;
  document.getElementById('labelLocal').textContent=`Victoria ${local}`;
  document.getElementById('labelVisit').textContent=`Victoria ${visita}`;
  document.getElementById('bLabelL').textContent=`Victoria ${local}`;
  document.getElementById('bLabelV').textContent=`Victoria ${visita}`;
  document.getElementById('probLocal').textContent=r.pL+'%';
  document.getElementById('probDraw').textContent=r.pE+'%';
  document.getElementById('probVisit').textContent=r.pV+'%';
  document.getElementById('bPL').textContent=r.pL+'%';
  document.getElementById('bPE').textContent=r.pE+'%';
  document.getElementById('bPV').textContent=r.pV+'%';

  setTimeout(()=>{
    document.getElementById('barLocal').style.width=r.pL+'%';
    document.getElementById('barDraw').style.width=r.pE+'%';
    document.getElementById('barVisit').style.width=r.pV+'%';
  },80);

  const lGrid=document.getElementById('lambdaGrid');
  lGrid.style.display='contents';
  lGrid.innerHTML=`<div class="lambda-g-cell"><div class="lambda-g-val">${r.ll}</div><span class="lambda-g-lab">λ ${local}</span></div><div class="lambda-g-cell"><div class="lambda-g-val">${r.lv}</div><span class="lambda-g-lab">λ ${visita}</span></div><div class="lambda-g-cell"><div class="lambda-g-val">${PROM_GOLES}</div><span class="lambda-g-lab">Prom. Liga</span></div>`;
  document.getElementById('lambdaSection').style.display='grid';

  document.getElementById('confPct').textContent=r.confianza+'%';
  document.getElementById('confBlock').style.display='flex';
  setTimeout(()=>{document.getElementById('confBar').style.width=r.confianza+'%';},80);

  drawPredChart(r.pL,r.pE,r.pV,local,visita);
  drawPoissonChart(r.ll,r.lv,local,visita);

  const sl=r.sl,sv=r.sv;
  document.getElementById('statsPanel').innerHTML=`
    <div class="stat-g"><div class="stat-g-val">${sl.gfpj.toFixed(2)}</div><div class="stat-g-lab">Goles a Favor / PJ</div><div class="stat-g-team">🏠 ${local}</div></div>
    <div class="stat-g"><div class="stat-g-val">${sv.gfpj.toFixed(2)}</div><div class="stat-g-lab">Goles a Favor / PJ</div><div class="stat-g-team">✈️ ${visita}</div></div>
    <div class="stat-g"><div class="stat-g-val">${sl.gcpj.toFixed(2)}</div><div class="stat-g-lab">Goles en Contra / PJ</div><div class="stat-g-team">🏠 ${local}</div></div>
    <div class="stat-g"><div class="stat-g-val">${sv.gcpj.toFixed(2)}</div><div class="stat-g-lab">Goles en Contra / PJ</div><div class="stat-g-team">✈️ ${visita}</div></div>
    <div class="stat-g"><div class="stat-g-val">${sl.fa.toFixed(3)}</div><div class="stat-g-lab">Fuerza de Ataque</div><div class="stat-g-team">🏠 ${local}</div></div>
    <div class="stat-g"><div class="stat-g-val">${sv.fa.toFixed(3)}</div><div class="stat-g-lab">Fuerza de Ataque</div><div class="stat-g-team">✈️ ${visita}</div></div>
    <div class="stat-g"><div class="stat-g-val">${sl.ir.toFixed(1)}</div><div class="stat-g-lab">Índice de Rendimiento</div><div class="stat-g-team">🏠 ${local}</div></div>
    <div class="stat-g"><div class="stat-g-val">${sv.ir.toFixed(1)}</div><div class="stat-g-lab">Índice de Rendimiento</div><div class="stat-g-team">✈️ ${visita}</div></div>`;

  const hist=historial(local,visita);
  document.getElementById('historyMatches').innerHTML=hist.map(h=>`<div class="history-g-row"><span>${h.fecha}</span><span>${h.score}</span></div>`).join('');
  document.getElementById('predictionResult').classList.add('active');
  document.getElementById('predictionResult').scrollIntoView({behavior:'smooth',block:'nearest'});
}

async function openModal(local,visita){
  const r=await predecir(local,visita);if(!r)return;
  document.getElementById('modalTitle').textContent=`${local} vs ${visita}`;
  document.getElementById('modalProbLocal').textContent=r.pL+'%';
  document.getElementById('modalProbEmpate').textContent=r.pE+'%';
  document.getElementById('modalProbVisit').textContent=r.pV+'%';

  modalChartDona=destroyChart(modalChartDona);
  modalChartDona=new Chart(document.getElementById('modalChartDona').getContext('2d'),{
    type:'doughnut',
    data:{labels:[`Victoria ${local}`,'Empate',`Victoria ${visita}`],datasets:[{data:[r.pL,r.pE,r.pV],backgroundColor:[C.blue,C.amber,C.lime],borderWidth:2}]},
    options:{responsive:true,plugins:{legend:{position:'bottom',labels:{color:C.tick,font:{size:10}}}},cutout:'55%'}
  });

  modalChartLambda=destroyChart(modalChartLambda);
  const sl=r.sl,sv=r.sv;
  modalChartLambda=new Chart(document.getElementById('modalChartLambda').getContext('2d'),{
    type:'bar',
    data:{labels:[local,visita],datasets:[{label:'λ (Goles esperados)',data:[r.ll,r.lv],backgroundColor:[C.blue,C.lime],borderRadius:6}]},
    options:{responsive:true,indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{beginAtZero:true,ticks:{color:C.tick},grid:{color:C.grid}},y:{ticks:{color:C.tick},grid:{display:false}}}}
  });

  document.getElementById('modalStats').innerHTML=`
    <div class="stat-item"><h5>Goles a Favor (prom.)</h5><div class="stat-value">${sl.gfpj.toFixed(2)} vs ${sv.gfpj.toFixed(2)}</div></div>
    <div class="stat-item"><h5>Goles en Contra (prom.)</h5><div class="stat-value">${sl.gcpj.toFixed(2)} vs ${sv.gcpj.toFixed(2)}</div></div>
    <div class="stat-item"><h5>Puntos por Partido</h5><div class="stat-value">${sl.ppg.toFixed(2)} vs ${sv.ppg.toFixed(2)}</div></div>
    <div class="stat-item"><h5>Índice de Rendimiento</h5><div class="stat-value">${sl.ir.toFixed(1)} vs ${sv.ir.toFixed(1)}</div></div>
    <div class="stat-item"><h5>Fuerza de Ataque</h5><div class="stat-value">${sl.fa.toFixed(3)} vs ${sv.fa.toFixed(3)}</div></div>
    <div class="stat-item"><h5>Fuerza de Defensa</h5><div class="stat-value">${sl.fd.toFixed(3)} vs ${sv.fd.toFixed(3)}</div></div>`;

  const dif=sl.ir-sv.ir,ventLocal=sl.ppg>sv.ppg?'Alta':'Media';
  document.getElementById('modalFactores').innerHTML=`
    <div class="history-match"><span>Ventaja de Localía</span><span style="color:#1a3a8f">+15% (Factor ${FAC_LOCAL}x)</span></div>
    <div class="history-match"><span>Diferencia IR</span><span style="color:${dif>0?'#1a3a8f':'#ef4444'}">${dif>0?'+':''}${dif.toFixed(1)}</span></div>
    <div class="history-match"><span>λ Local vs Visitante</span><span style="font-weight:700">${r.ll} vs ${r.lv}</span></div>
    <div class="history-match"><span>Forma Reciente</span><span style="color:#84cc16;font-weight:700">${ventLocal}</span></div>`;

  const hist=historial(local,visita);
  document.getElementById('modalHistorial').innerHTML=hist.map(h=>`<div class="history-match"><span>${h.fecha}</span><span>${h.score}</span></div>`).join('');
  document.getElementById('detailModal').classList.add('active');
  document.body.style.overflow='hidden';
}

function closeModal(){
  document.getElementById('detailModal').classList.remove('active');
  document.body.style.overflow='auto';
}

function updateTeamLogo(selId,logoId){
  const sel=document.getElementById(selId),cont=document.getElementById(logoId),eq=sel.value;
  if(eq&&LOGOS[eq]){cont.innerHTML=`<img src="${LOGOS[eq]}" alt="${eq}">`;cont.classList.remove('empty');}
  else{cont.innerHTML='<span>⚽</span>';cont.classList.add('empty');}
}

window.addEventListener('click',e=>{if(e.target===document.getElementById('detailModal'))closeModal();});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal();});
window.addEventListener('DOMContentLoaded', cargarDatos);
