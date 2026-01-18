/* Shared JS for the Transformer/LLM interactive study site.
   - lightweight: vanilla JS, no build tools
   - demos are opt-in via data-demo attributes
*/

function $(sel, root = document){ return root.querySelector(sel); }
function $all(sel, root = document){ return Array.from(root.querySelectorAll(sel)); }

function safeJsonParse(text){
  try { return { ok:true, value: JSON.parse(text) }; }
  catch (e){ return { ok:false, error: String(e) }; }
}

function fmt(x){
  if (typeof x === 'number'){
    const v = Math.abs(x) < 1e-10 ? 0 : x;
    return Number.isFinite(v) ? (Math.round(v * 1e6) / 1e6) : v;
  }
  return x;
}

function matmul(A, B){
  const m = A.length, k = A[0].length, n = B[0].length;
  const out = Array.from({length:m}, () => Array(n).fill(0));
  for (let i=0;i<m;i++){
    for (let t=0;t<k;t++){
      const a = A[i][t];
      for (let j=0;j<n;j++){
        out[i][j] += a * B[t][j];
      }
    }
  }
  return out;
}

function transpose(A){
  const m=A.length, n=A[0].length;
  const out=Array.from({length:n},()=>Array(m).fill(0));
  for (let i=0;i<m;i++) for (let j=0;j<n;j++) out[j][i]=A[i][j];
  return out;
}

function softmaxRowWise(A){
  // A: (S,S) or (S,T)
  return A.map(row=>{
    const max = Math.max(...row);
    const exps = row.map(v=>Math.exp(v-max));
    const sum = exps.reduce((a,b)=>a+b,0);
    return exps.map(v=>v/sum);
  });
}

function scale(A, s){
  return A.map(r=>r.map(v=>v*s));
}

function elemwise(A, f){
  return A.map(r=>r.map(f));
}

function attention(Q, K, V){
  // Q: (S, dk), K: (S, dk), V: (S, dv)
  const dk = Q[0].length;
  const scores = scale(matmul(Q, transpose(K)), 1/Math.sqrt(dk));
  const probs = softmaxRowWise(scores);
  const out = matmul(probs, V);
  return { scores, probs, out };
}

function initAttentionDemo(root){
  const qEl = $('[data-role="Q"]', root);
  const kEl = $('[data-role="K"]', root);
  const vEl = $('[data-role="V"]', root);
  const outEl = $('[data-role="OUT"]', root);
  const btn = $('[data-action="run"]', root);
  const btnReset = $('[data-action="reset"]', root);

  const defaults = {
    Q: [[1,0],[0,1],[1,1]],
    K: [[1,0],[0,1],[1,1]],
    V: [[1,2],[3,4],[5,6]]
  };

  function reset(){
    qEl.value = JSON.stringify(defaults.Q, null, 2);
    kEl.value = JSON.stringify(defaults.K, null, 2);
    vEl.value = JSON.stringify(defaults.V, null, 2);
    outEl.textContent = '点击“运行”以计算 Attention(Q,K,V)。';
  }

  function run(){
    const q = safeJsonParse(qEl.value);
    const k = safeJsonParse(kEl.value);
    const v = safeJsonParse(vEl.value);
    if (!q.ok || !k.ok || !v.ok){
      outEl.textContent = `JSON 解析失败：\nQ: ${q.ok?'OK':q.error}\nK: ${k.ok?'OK':k.error}\nV: ${v.ok?'OK':v.error}`;
      return;
    }
    try{
      const {scores, probs, out} = attention(q.value, k.value, v.value);
      const pretty = (M)=>JSON.stringify(elemwise(M, fmt), null, 2);
      outEl.textContent =
        '1) 分数矩阵 Scores = QK^T / sqrt(d_k)\n' + pretty(scores) + '\n\n' +
        '2) 注意力权重 Probs = softmax(Scores)\n' + pretty(probs) + '\n\n' +
        '3) 输出 Out = Probs * V\n' + pretty(out);
    }catch(e){
      outEl.textContent = '计算失败：' + String(e);
    }
  }

  btn?.addEventListener('click', run);
  btnReset?.addEventListener('click', reset);
  reset();
}

function initPosEncDemo(root){
  const posEl = $('[data-role="pos"]', root);
  const dimEl = $('[data-role="dim"]', root);
  const outEl = $('[data-role="OUT"]', root);
  const btn = $('[data-action="run"]', root);
  const canvas = $('canvas', root);
  const ctx = canvas?.getContext?.('2d');

  function sinCosPE(pos, dModel){
    const pe = Array(dModel).fill(0);
    for (let i=0;i<dModel;i++){
      const k = Math.floor(i/2);
      const denom = Math.pow(10000, (2*k)/dModel);
      const angle = pos/denom;
      pe[i] = (i%2===0) ? Math.sin(angle) : Math.cos(angle);
    }
    return pe;
  }

  function drawSeries(series){
    if (!ctx || !canvas) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0,0,W,H);
    // axes
    ctx.globalAlpha = 1;
    ctx.strokeStyle = 'rgba(255,255,255,.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10);
    ctx.lineTo(40, H-26);
    ctx.lineTo(W-10, H-26);
    ctx.stroke();

    const plotW = W-55, plotH = H-40;
    const minY = -1, maxY = 1;
    const n = series.length;
    function x(i){ return 40 + (i/(n-1))*plotW; }
    function y(v){
      const t = (v-minY)/(maxY-minY);
      return (H-26) - t*plotH;
    }
    // line
    ctx.strokeStyle = 'rgba(122,162,255,.95)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((v,i)=>{
      const X = x(i), Y = y(v);
      if (i===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y);
    });
    ctx.stroke();
  }

  function run(){
    const pos = Number(posEl.value);
    const d = Number(dimEl.value);
    if (!Number.isFinite(pos) || !Number.isFinite(d) || d<=0){
      outEl.textContent = '请输入合法的 pos 与 d_model。';
      return;
    }
    const pe = sinCosPE(pos, d);
    outEl.textContent = JSON.stringify(pe.map(fmt), null, 2);
    // draw first 64 dims (or all if smaller)
    drawSeries(pe.slice(0, Math.min(64, pe.length)));
  }

  btn?.addEventListener('click', run);
  // default
  posEl.value = '10';
  dimEl.value = '64';
  run();
}

function initQuiz(root){
  const checks = $all('input[type="radio"]', root);
  const btn = $('[data-action="grade"]', root);
  const out = $('[data-role="OUT"]', root);
  btn?.addEventListener('click', ()=>{
    const groups = new Map();
    checks.forEach(r=>{
      const name=r.name;
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(r);
    });
    let total=0, ok=0;
    for (const [name, radios] of groups.entries()){
      total += 1;
      const chosen = radios.find(r=>r.checked);
      if (chosen && chosen.dataset.correct === '1') ok += 1;
    }
    out.textContent = `得分：${ok}/${total}。` + (ok===total ? '（全对）' : '（可回到正文对照公式/定义再试一次）');
  });
}

function initKVCalcDemo(root){
  const getNum = (role) => Number($(`[data-role="${role}"]`, root)?.value ?? NaN);
  const out = $('[data-role="OUT"]', root);
  const btn = $('[data-action="run"]', root);

  function bytesToHuman(n){
    const units = ['B','KB','MB','GB','TB'];
    let v = n, i = 0;
    while (v >= 1024 && i < units.length-1){
      v /= 1024; i += 1;
    }
    return `${(Math.round(v*100)/100)} ${units[i]}`;
  }

  function run(){
    const B = getNum('B');
    const L = getNum('L');
    const H = getNum('H');
    const D = getNum('D');
    const S = getNum('S');
    const bytes = getNum('bytes');
    if (![B,L,H,D,S,bytes].every(Number.isFinite) || Math.min(B,L,H,D,S,bytes) <= 0){
      out.textContent = '请输入全为正数的参数。';
      return;
    }
    const kvBytes = 2 * B * L * H * S * D * bytes; // 2 for K+V
    const perToken = 2 * B * L * H * D * bytes;    // incremental per new token
    out.textContent =
      `KV cache 总大小（近似）：${bytesToHuman(kvBytes)}\\n` +
      `每增加 1 个 token 增量：${bytesToHuman(perToken)}\\n\\n` +
      `公式：2*(K+V)*B*L*H*S*D*bytes = 2*${B}*${L}*${H}*${S}*${D}*${bytes}`;
  }

  btn?.addEventListener('click', run);
  run();
}

function initAll(){
  // mark active nav link
  const here = location.pathname.split('/').pop();
  $all('.navlinks a').forEach(a=>{
    const href = a.getAttribute('href') || '';
    if (href.endsWith(here)) a.classList.add('active');
  });
  $all('[data-demo="attention"]').forEach(initAttentionDemo);
  $all('[data-demo="posenc"]').forEach(initPosEncDemo);
  $all('[data-demo="kvcalc"]').forEach(initKVCalcDemo);
  $all('[data-demo="quiz"]').forEach(initQuiz);
}

document.addEventListener('DOMContentLoaded', initAll);

