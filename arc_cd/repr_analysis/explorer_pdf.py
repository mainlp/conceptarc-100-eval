"""Offline vector PDF export for the explorer's SVG scatter plot."""

EXPLORER_PDF_SCRIPT = r"""
function exportPlotPdf() {
  const plot = byId('plot'), view = plot.viewBox.baseVal;
  const width = 900, height = 570, scale = width / view.width;
  const screenScale = Math.hypot(plot.getScreenCTM().a,plot.getScreenCTM().b);
  const format = value => Number(value.toFixed(6)).toString();
  const objects = [], addObject = value => {objects.push(value); return objects.length;};
  const stream = (dictionary,content) => addObject(`<< ${dictionary} /Length ${content.length} >>\nstream\n${content}\nendstream`);
  const colorCanvas = document.createElement('canvas'); colorCanvas.width = colorCanvas.height = 1;
  const context = colorCanvas.getContext('2d',{willReadFrequently:true}), colorCache = new Map();
  function rgb(color) {
    if(!colorCache.has(color)) {
      context.clearRect(0,0,1,1); context.fillStyle = color; context.fillRect(0,0,1,1);
      colorCache.set(color,[...context.getImageData(0,0,1,1).data].slice(0,3).map(value => format(value/255)).join(' '));
    }
    return colorCache.get(color);
  }
  function circlePath(radius) {
    const r = format(radius), k = format(radius*0.5522847498307936);
    return `${r} 0 m ${r} ${k} ${k} ${r} 0 ${r} c -${k} ${r} -${r} ${k} -${r} 0 c -${r} -${k} -${k} -${r} 0 -${r} c ${k} -${r} ${r} -${k} ${r} 0 c h`;
  }
  const catalog = addObject(''), pages = addObject(''), page = addObject('');
  const forms = new Map(), states = new Map(), commands = [`q 0 0 ${width} ${height} re W n`,`${rgb(getComputedStyle(plot).backgroundColor)} rg 0 0 ${width} ${height} re f`];
  for(const circle of byId('points').children) {
    const style = getComputedStyle(circle);
    if(style.display === 'none' || style.visibility === 'hidden') continue;
    const x = (circle.cx.baseVal.value-view.x)*scale, y = height-(circle.cy.baseVal.value-view.y)*scale;
    const radius = circle.r.baseVal.value*scale, strokeWidth = parseFloat(style.strokeWidth)*scale/screenScale;
    const extent = radius+strokeWidth/2;
    if(x+extent < 0 || x-extent > width || y+extent < 0 || y-extent > height) continue;
    const opacity = Number(style.opacity);
    if(!states.has(opacity)) states.set(opacity,{name:`GS${states.size}`,id:addObject(`<< /Type /ExtGState /ca ${format(opacity)} /CA ${format(opacity)} >>`)});
    const fill = circle.getAttribute('fill'), stroke = style.stroke, key = JSON.stringify([fill,stroke,radius,strokeWidth]);
    if(!forms.has(key)) {
      const path = circlePath(radius), drawing = [];
      if(fill.startsWith('url(#')) {
        // Family gradients have repeated stops: preserve their hard color bands.
        const stops = [...byId(fill.slice(5,-1)).children];
        drawing.push(`q ${path} W n`);
        for(let index=0; index+1<stops.length; index++) {
          const start = stops[index].offset.baseVal, end = stops[index+1].offset.baseVal;
          if(end <= start) continue;
          drawing.push(`${rgb(getComputedStyle(stops[index]).stopColor)} rg ${format((2*start-1)*radius)} ${format(-radius)} ${format(2*(end-start)*radius)} ${format(2*radius)} re f`);
        }
        drawing.push('Q');
      } else if(fill !== 'none') drawing.push(`${rgb(style.fill)} rg ${path} f`);
      if(stroke !== 'none' && strokeWidth > 0) drawing.push(`${rgb(stroke)} RG ${format(strokeWidth)} w ${path} S`);
      // Isolate fill and stroke so selection opacity applies to the whole point.
      const bounds = `${format(-extent)} ${format(-extent)} ${format(extent)} ${format(extent)}`;
      const id = stream(`/Type /XObject /Subtype /Form /BBox [${bounds}] /Resources << >> /Group << /S /Transparency /CS /DeviceRGB /I true >>`,drawing.join('\n'));
      forms.set(key,{name:`P${forms.size}`,id});
    }
    commands.push(`q /${states.get(opacity).name} gs 1 0 0 1 ${format(x)} ${format(y)} cm /${forms.get(key).name} Do Q`);
  }
  commands.push('Q');
  const contents = stream('',commands.join('\n'));
  const resources = entries => [...entries.values()].map(entry => `/${entry.name} ${entry.id} 0 R`).join(' ');
  objects[catalog-1] = `<< /Type /Catalog /Pages ${pages} 0 R >>`;
  objects[pages-1] = `<< /Type /Pages /Kids [${page} 0 R] /Count 1 >>`;
  objects[page-1] = `<< /Type /Page /Parent ${pages} 0 R /MediaBox [0 0 ${width} ${height}] /Resources << /XObject << ${resources(forms)} >> /ExtGState << ${resources(states)} >> >> /Contents ${contents} 0 R >>`;
  // All PDF syntax above is ASCII, so string lengths are byte offsets.
  let pdf = '%PDF-1.4\n', offsets = [0];
  objects.forEach((object,index) => {offsets.push(pdf.length); pdf += `${index+1} 0 obj\n${object}\nendobj\n`;});
  const xref = pdf.length;
  pdf += `xref\n0 ${objects.length+1}\n0000000000 65535 f \n`;
  for(const offset of offsets.slice(1)) pdf += `${String(offset).padStart(10,'0')} 00000 n \n`;
  pdf += `trailer\n<< /Size ${objects.length+1} /Root ${catalog} 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  const url = URL.createObjectURL(new Blob([pdf],{type:'application/pdf'})), link = element('a');
  const parts = [data.model,axes.population?.name,byId('method').value,frame.position,`layer_${frame.layer}`,byId('color').value].filter(Boolean);
  link.href = url; link.download = parts.join('_').replace(/[^a-zA-Z0-9_.-]+/g,'_')+'.pdf';
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
}
byId('export-pdf').addEventListener('click', () => {
  const button = byId('export-pdf'), status = byId('pdf-status');
  button.disabled = true; status.textContent = '';
  try {exportPlotPdf();} catch(error) {status.textContent = 'PDF export failed. Please try again.'; console.error(error);}
  finally {button.disabled = false;}
});
"""
