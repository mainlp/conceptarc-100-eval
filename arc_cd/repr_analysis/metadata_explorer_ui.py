"""Optional metadata controls embedded into the shared offline explorer."""

METADATA_CSS = r"""
[hidden]{display:none!important}input[type=search]{width:100%;padding:7px;margin:5px 0;border:1px solid #b9c9d4;border-radius:6px;font:inherit;color:var(--ink)}
.with-metadata aside{max-height:calc(100vh - 155px);overflow-y:auto;scrollbar-gutter:stable}.with-metadata #groups{max-height:220px}.with-metadata #legend{max-height:110px;overflow-y:auto}
#metadata-panel label{margin-top:10px}#tag-value{font-size:12px}#metadata-panel .buttons{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}#active-filters{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}#active-filters button{font-size:11px;overflow-wrap:anywhere;text-align:left}
#composition-panel{margin-top:16px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;font-weight:600}#composition-panel select{margin-bottom:8px}#composition-scroll{max-height:210px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:6px;text-align:left;border-bottom:1px solid var(--line);overflow-wrap:anywhere}th{background:#f5f7f9;position:sticky;top:0}td:first-child{max-width:220px}td:not(:first-child){white-space:nowrap}
.task-description{font-size:13px;line-height:1.5}.task-metadata{margin:10px 0;font-size:12px}.task-metadata dl{margin:8px 0}.task-metadata dt{font-weight:600;overflow-wrap:anywhere;margin-top:7px}.task-metadata dd{margin:3px 0;overflow-wrap:anywhere}.metadata-tags{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0}.metadata-tags button{font-size:11px;padding:4px 6px}.task-metadata pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;max-height:300px;overflow:auto}.metadata-source{overflow-wrap:anywhere}
@media(max-width:1100px){.with-metadata aside{max-height:none;overflow:visible;scrollbar-gutter:auto}}
"""

METADATA_CONTROLS = r"""
<section class="panel" id="metadata-panel" hidden><h2>Task metadata</h2>
<p class="small">Declared spec properties, shared by all eight variants. Branch tags describe possible actions.</p>
<label>Concepts · all if empty<select id="concept-filter" multiple size="3" aria-label="Concept filter"></select></label>
<label>Metadata field<input type="search" id="field-search" placeholder="Find a field" aria-label="Search metadata fields"><select id="tag-field" aria-label="Metadata field"></select></label>
<label>Value to highlight<input type="search" id="tag-search" placeholder="Find a value" aria-label="Search metadata values"><select id="tag-value" size="4" aria-label="Metadata value"></select></label>
<div class="buttons"><button id="add-filter" disabled>Filter to value</button><button id="clear-highlight">Clear highlight</button><button id="clear-filters">Clear filters</button></div>
<p class="small">Filters use OR within a field and AND across fields.</p><div id="active-filters"></div><p id="filter-status" class="small" aria-live="polite"></p></section>
"""

COMPOSITION_HTML = r"""
<details id="composition-panel" hidden open><summary>Selection metadata</summary>
<p class="small">Distinct parent tasks per tag. Tags can overlap, so percentages may sum above 100%.</p>
<label>Compare field<select id="composition-field"></select></label>
<div id="composition-scroll"><table><thead><tr><th>Value</th><th>Selected</th><th>Visible population</th></tr></thead><tbody id="composition-body"></tbody></table></div></details>
"""

METADATA_SCRIPT = r"""
const metadata = data.metadata || null;
const metadataFields = new Map((metadata?.fields || []).map(field => [field.id,field]));
const metadataFilters = new Map();
const familyPaints = new Map();
let conceptFilters = new Set(), visibleSet = new Set(allTasks()), highlightedTag = null;
function visibleTasks() {return [...visibleSet];}
function taskVisible(task) {return visibleSet.has(task);}
function hasValue(task, field, value) {return (metadata.records[task].values[field] || []).includes(value);}
function valueLabel(field, value) {return metadataFields.get(field)?.values.find(entry => entry.value === value)?.label ?? value;}
function categoricalColor(field, value) {
  const entries = metadataFields.get(field)?.values || [], index = entries.findIndex(entry => entry.value === value);
  return index < 0 ? '#bdc5cc' : `hsl(${(index * 137.508) % 360} 60% 42%)`;
}
function familyPointColor(task, field) {
  const values = metadata.records[task].values[field] || [];
  if(!values.length) return '#bdc5cc';
  if(values.length === 1) return categoricalColor(field,values[0]);
  const key = JSON.stringify([field,values]);
  if(!familyPaints.has(key)) {
    const gradient = document.createElementNS(ns,'linearGradient'), id = `family-colors-${familyPaints.size}`;
    for(const [name,value] of Object.entries({id,x1:'0%',x2:'100%',y1:'0%',y2:'0%',gradientUnits:'objectBoundingBox'})) gradient.setAttribute(name,value);
    values.forEach((value,index) => {
      for(const boundary of [index,index+1]) {
        const stop = document.createElementNS(ns,'stop'); stop.setAttribute('offset',`${100*boundary/values.length}%`); stop.setAttribute('stop-color',categoricalColor(field,value)); gradient.append(stop);
      }
    });
    byId('metadata-color-defs').append(gradient); familyPaints.set(key,`url(#${id})`);
  }
  return familyPaints.get(key);
}
function pointDescription(task) {
  const base = `${axes.task_ids[task]} · ${clusterName(frame.cluster_labels[task])}`, mode = byId('color').value;
  if(!metadata || !(mode.startsWith('family.') || mode.startsWith('op.'))) return base;
  const labels = (metadata.records[task].values[mode] || []).map(value => valueLabel(mode,value));
  return `${base} · ${metadataFields.get(mode)?.label || mode}: ${labels.join(', ') || 'Unspecified / unmapped'}`;
}
function metadataPointColor(task) {
  const mode = byId('color').value;
  if(mode === 'metadata_tag') return highlightedTag && hasValue(task,highlightedTag.field,highlightedTag.value) ? '#b44818' : '#bdc5cc';
  if(mode.startsWith('family.')) return familyPointColor(task,mode);
  if(mode.startsWith('op.')) return categoricalColor(mode,(metadata.records[task].values[mode] || [])[0]);
  return mode === 'cluster' ? clusterColor(frame.cluster_labels[task]) : colors[axes.labels[task] % colors.length];
}
function highlightTag(field, value) {
  highlightedTag = {field,value};
  if(byId('tag-field').value !== field || ![...byId('tag-value').options].some(option => option.value === value)) {byId('field-search').value = ''; byId('tag-search').value = ''; renderFieldChoices(); byId('tag-field').value = field;}
  renderTagValues(); byId('color').value = 'metadata_tag';
  updateHighlights(); renderLegend(); renderFilterStatus();
}
function renderFilterStatus() {
  if(!metadata) return;
  const count = highlightedTag ? visibleTasks().filter(task => hasValue(task,highlightedTag.field,highlightedTag.value)).length : null;
  byId('filter-status').textContent = `${visibleSet.size} / ${axes.task_ids.length} tasks visible` + (count !== null ? ` · ${count} match ${metadataFields.get(highlightedTag.field).label}: ${valueLabel(highlightedTag.field,highlightedTag.value)}` : '');
}
function applyMetadataFilters() {
  visibleSet = new Set(allTasks().filter(task => (!conceptFilters.size || conceptFilters.has(axes.labels[task])) && [...metadataFilters].every(([field,values]) => [...values].some(value => hasValue(task,field,value)))));
  const chips = byId('active-filters'); chips.replaceChildren();
  for(const [field,values] of metadataFilters) for(const value of values) {
    const button = element('button',`${metadataFields.get(field).label}: ${valueLabel(field,value)} ×`);
    button.dataset.field = field; button.dataset.value = value;
    button.addEventListener('click', () => {values.delete(value); if(!values.size) metadataFilters.delete(field); applyMetadataFilters();}); chips.append(button);
  }
  renderGroups(); renderLegend(); renderTagValues(); renderFilterStatus(); setSelection(visibleTasks(),metadataFilters.size || conceptFilters.size ? 'Filtered tasks' : 'All tasks');
}
function renderFieldChoices() {
  const select = byId('tag-field'), current = select.value, query = byId('field-search').value.toLowerCase(); select.replaceChildren();
  for(const field of metadata.fields) if(`${field.label} ${field.id}`.toLowerCase().includes(query)) {const option = element('option',field.label); option.value = field.id; select.append(option);}
  if([...select.options].some(option => option.value === current)) select.value = current;
  renderTagValues();
}
function renderTagValues() {
  const select = byId('tag-value'), field = metadataFields.get(byId('tag-field').value), query = byId('tag-search').value.toLowerCase(); select.replaceChildren();
  if(field) for(const entry of field.values) if(`${entry.label} ${entry.description}`.toLowerCase().includes(query)) {
    const count = visibleTasks().filter(task => hasValue(task,field.id,entry.value)).length;
    const option = element('option',`${entry.label} (${count})`); option.value = entry.value; option.title = entry.description; select.append(option);
  }
  select.selectedIndex = -1;
  if(highlightedTag && highlightedTag.field === field?.id && [...select.options].some(option => option.value === highlightedTag.value)) select.value = highlightedTag.value;
  byId('add-filter').disabled = select.selectedIndex < 0;
}
function renderComposition() {
  if(!metadata) return;
  const field = metadataFields.get(byId('composition-field').value), body = byId('composition-body'); body.replaceChildren();
  if(!field) return;
  const visible = visibleTasks(), rows = field.values.map(entry => ({...entry, selected:selected.filter(task => hasValue(task,field.id,entry.value)).length, visible:visible.filter(task => hasValue(task,field.id,entry.value)).length}));
  rows.push({value:null,label:'Unspecified / unmapped',selected:selected.filter(task => !(metadata.records[task].values[field.id] || []).length).length,visible:visible.filter(task => !(metadata.records[task].values[field.id] || []).length).length});
  rows.sort((a,b) => b.selected-a.selected || b.visible-a.visible || a.label.localeCompare(b.label));
  function cell(count,total) {return `${count} / ${total} (${total ? (100*count/total).toFixed(1)+'%' : '—'})`;}
  for(const entry of rows) if(entry.visible || entry.selected) {
    const row = element('tr'); row.dataset.value = entry.value ?? '__missing__'; row.dataset.selected = entry.selected; row.dataset.visible = entry.visible;
    row.append(element('td',entry.label),element('td',cell(entry.selected,selected.length)),element('td',cell(entry.visible,visible.length))); body.append(row);
  }
  if(!rows.some(row => row.visible)) {const row = element('tr'), cell = element('td','No visible tasks'); cell.colSpan = 3; row.append(cell); body.append(row);}
}
function renderTaskMetadata(task, card) {
  if(!metadata) return;
  const record = metadata.records[task]; card.append(element('p',record.description,'task-description'));
  const details = element('details',undefined,'task-metadata'); details.append(element('summary','Task specification'));
  const roots = element('dl'); for(const role of ['layout','selector','action']) roots.append(element('dt',role),element('dd',record.root_ops[role])); details.append(roots);
  const tags = element('div',undefined,'metadata-tags');
  for(const role of ['layout','selector','action']) for(const family of record.families[role]) {
    const field = `family.${role}`, button = element('button',`${role}: ${valueLabel(field,family)}`); button.title = metadata.taxonomy.families[role][family].description;
    button.addEventListener('click', () => highlightTag(field,family)); tags.append(button);
  } details.append(tags);
  const params = element('details'); params.append(element('summary',`Parameters (${record.parameters.length})`));
  const list = element('dl'); for(const parameter of record.parameters) list.append(element('dt',parameter.path.replace(/^program\./,'')),element('dd',JSON.stringify(parameter.value))); params.append(list); details.append(params);
  const evidence = element('details'); evidence.append(element('summary','Operation paths and mapping evidence'));
  if(record.action_family_evidence) evidence.append(element('p',`Action family source: declared concept ${record.action_family_evidence.value}.`,'small'));
  for(const node of record.operations) {
    evidence.append(element('p',`${node.path} · ${node.role}: ${node.op} · ${node.status}${node.families.length ? ' → '+node.families.join(', ') : ''}`,'small'));
    if(node.secondary_families?.length) evidence.append(element('p',`Secondary operation details: ${node.secondary_families.map(family => metadata.taxonomy.families.action[family]?.label ?? family).join(', ')}.`,'small'));
    for(const rule of node.parameter_evidence || []) evidence.append(element('p',`${rule.path} = ${JSON.stringify(rule.value)} → ${rule.family}: ${rule.description}`,'small'));
  } details.append(evidence);
  const raw = element('details',undefined,'raw-program'); raw.append(element('summary','Raw program'),element('pre',JSON.stringify(record.program,null,2))); details.append(raw);
  details.append(element('p',`Taxonomy ${metadata.taxonomy_version} · ${record.spec_path}`,'small metadata-source')); card.append(details);
}
function initializeMetadata() {
  if(!metadata) return;
  document.body.classList.add('with-metadata'); byId('metadata-panel').hidden = false; byId('composition-panel').hidden = false; byId('legend-search').hidden = false;
  for(const role of ['layout','selector','action']) {const option = element('option',`${role[0].toUpperCase()+role.slice(1)} operation`); option.value = `op.${role}`; byId('color').append(option);}
  for(const role of ['action','layout','selector']) {const option = element('option',`${role[0].toUpperCase()+role.slice(1)} family`); option.value = `family.${role}`; byId('color').append(option);}
  const highlight = element('option','Metadata highlight'); highlight.value = 'metadata_tag'; byId('color').append(highlight);
  axes.concepts.forEach((name,index) => {const option = element('option',name); option.value = index; byId('concept-filter').append(option);});
  for(const field of metadata.fields) {const option = element('option',field.label); option.value = field.id; byId('composition-field').append(option);}
  renderFieldChoices(); byId('tag-field').value = metadataFields.has('family.action') ? 'family.action' : metadata.fields[0]?.id;
  byId('composition-field').value = byId('tag-field').value; renderTagValues();
  byId('field-search').addEventListener('input',renderFieldChoices); byId('tag-search').addEventListener('input',renderTagValues); byId('tag-field').addEventListener('change',renderTagValues);
  byId('tag-value').addEventListener('change', () => {if(byId('tag-value').selectedIndex >= 0) highlightTag(byId('tag-field').value,byId('tag-value').value);});
  byId('concept-filter').addEventListener('change', () => {conceptFilters = new Set([...byId('concept-filter').selectedOptions].map(option => Number(option.value))); applyMetadataFilters();});
  byId('add-filter').addEventListener('click', () => {const select = byId('tag-value'); if(select.selectedIndex < 0) return; const field = byId('tag-field').value, value = select.value; if(!metadataFilters.has(field)) metadataFilters.set(field,new Set()); metadataFilters.get(field).add(value); applyMetadataFilters();});
  byId('clear-highlight').addEventListener('click', () => {highlightedTag = null; byId('add-filter').disabled = true; byId('tag-value').selectedIndex = -1; byId('color').value = 'concept'; updateHighlights(); renderLegend(); renderFilterStatus();});
  byId('clear-filters').addEventListener('click', () => {metadataFilters.clear(); conceptFilters.clear(); for(const option of byId('concept-filter').options) option.selected = false; applyMetadataFilters();});
  byId('composition-field').addEventListener('change',renderComposition); byId('legend-search').addEventListener('input',renderLegend);
}
"""
