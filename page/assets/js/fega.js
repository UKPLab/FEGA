const familyData = {
  directional: { label: 'Directional', title: 'One dominant effect direction', copy: 'Effects line up around a common axis. This is the closest match to a feature behaving like a stable, reusable steering direction.', structure: 'High', context: 'Low', points: [[125,242],[160,227],[195,212],[230,202],[265,186],[300,177],[335,158],[370,148],[405,133],[440,117],[475,107],[510,92]], arrow: [115,250,520,85] },
  subspace: { label: 'Low-dimensional', title: 'Several shared degrees of freedom', copy: 'Effects occupy a compact plane or subspace rather than one line. The feature has structure, but its outcome depends on more than one mode.', structure: 'Moderate', context: 'Moderate', points: [[140,245],[177,206],[206,236],[234,177],[263,215],[290,150],[318,194],[347,125],[375,170],[408,103],[440,148],[475,85]], arrow: [135,248,475,85] },
  clustered: { label: 'Clustered', title: 'Distinct context-specific modes', copy: 'Effects form separated groups. The same feature may participate in different behaviors depending on the context in which it is active.', structure: 'Grouped', context: 'High', points: [[160,238],[177,224],[190,245],[202,230],[180,251],[370,125],[386,112],[400,138],[418,119],[430,143],[462,158],[450,132]], arrow: null },
  diffuse: { label: 'Diffuse', title: 'No compact shared geometry', copy: 'Effects spread broadly through output space. This pattern is consistent with highly context-dependent operations, such as pointer-like behavior.', structure: 'Low', context: 'High', points: [[125,257],[158,145],[185,224],[222,108],[245,250],[279,173],[315,92],[340,231],[380,133],[412,270],[447,101],[482,190],[515,144]], arrow: null }
};
const plotPoints = document.getElementById('plot-points');
function drawFamily(name) {
  const data = familyData[name];
  document.getElementById('plot-family').textContent = data.label;
  document.getElementById('insight-title').textContent = data.title;
  document.getElementById('insight-copy').textContent = data.copy;
  document.getElementById('signal-structure').textContent = data.structure;
  document.getElementById('signal-context').textContent = data.context;
  document.getElementById('plot-description').textContent = data.title;
  plotPoints.replaceChildren();
  if (data.arrow) { const line = document.createElementNS('http://www.w3.org/2000/svg','line'); line.setAttribute('class','arrow'); line.setAttribute('x1',data.arrow[0]); line.setAttribute('y1',data.arrow[1]); line.setAttribute('x2',data.arrow[2]); line.setAttribute('y2',data.arrow[3]); plotPoints.append(line); }
  data.points.forEach(([x,y], index) => { const point = document.createElementNS('http://www.w3.org/2000/svg','circle'); point.setAttribute('class','point'); point.setAttribute('cx',x); point.setAttribute('cy',y); point.setAttribute('r', index % 3 === 0 ? 6 : 4.5); plotPoints.append(point); });
}
document.querySelectorAll('.geometry-choice').forEach(button => button.addEventListener('click', () => { document.querySelectorAll('.geometry-choice').forEach(item => { item.classList.remove('is-active'); item.setAttribute('aria-pressed','false'); }); button.classList.add('is-active'); button.setAttribute('aria-pressed','true'); drawFamily(button.dataset.family); }));
document.getElementById('copy-bibtex').addEventListener('click', async event => { const button = event.currentTarget; try { await navigator.clipboard.writeText(document.getElementById('bibtex').textContent); button.innerHTML = '<i class="fas fa-check"></i><span>Copied</span>'; setTimeout(() => button.innerHTML = '<i class="far fa-copy"></i><span>Copy</span>', 1600); } catch { button.querySelector('span').textContent = 'Select text'; } });
const scrollTop = document.getElementById('scroll-top'); window.addEventListener('scroll', () => scrollTop.classList.toggle('visible', window.scrollY > 400)); scrollTop.addEventListener('click', () => window.scrollTo({top: 0, behavior: 'smooth'}));
drawFamily('directional');
