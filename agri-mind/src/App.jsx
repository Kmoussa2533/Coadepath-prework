import React, { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import Papa from "papaparse";
/*import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Area, AreaChart, Legend } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Upload, Video, Image as ImageIcon, LineChart as LineChartIcon, Settings, Leaf, Droplets, Tractor, Sun, CloudRain, Play, Pause, Wand2, MapPinned } from "lucide-react";
*/
function parseCsv(fileOrText) {
  return new Promise((resolve, reject) => {
    if (!fileOrText) return resolve([]);
    if (typeof fileOrText === "string") {
      Papa.parse(fileOrText, { header: true, dynamicTyping: true, skipEmptyLines: true, complete: res => resolve(res.data), error: reject });
    } else {
      Papa.parse(fileOrText, { header: true, dynamicTyping: true, skipEmptyLines: true, complete: res => resolve(res.data), error: reject });
    }
  });
}

function toChartData(rows) {
  if (!rows || rows.length === 0) return [];
  return rows.map((r) => ({
    date: r.date || r.Date || r.timestamp || r.Timestamp || r.year || r.Year || "",
    yield: parseNum(r.yield ?? r.Yield),
    rainfall: parseNum(r.rainfall ?? r.Rainfall),
    temp: parseNum(r.temp ?? r.Temp ?? r.temperature ?? r.Temperature),
    ndvi: parseNum(r.ndvi ?? r.NDVI),
  }));
}
function parseNum(x){ if (typeof x === 'number') return x; if (x == null || x === '') return undefined; const n = Number(x); return Number.isFinite(n) ? n : undefined; }

function simpleMovingAverage(arr, k=3){ if(!arr || arr.length===0) return []; const out=[]; for(let i=0;i<arr.length;i++){ const s=Math.max(0,i-k+1), e=i+1; const slice=arr.slice(s,e); const vals=slice.filter((v)=>Number.isFinite(v)); out.push(vals.length? vals.reduce((a,b)=>a+b,0)/vals.length : undefined); } return out; }
function scaleTo255(value, min, max){ if(!Number.isFinite(value) || !Number.isFinite(min) || !Number.isFinite(max)) return 0; if(max===min) return 0; return Math.max(0, Math.min(255, Math.round(((value-min)/(max-min))*255))); }

export default function App(){
  const [dark, setDark] = useState(true);
  const [rows, setRows] = useState([]);
  const [chartData, setChartData] = useState([]);
  const [modelStatus, setModelStatus] = useState("Idle");
  const [forecast, setForecast] = useState([]);
  const [recommendations, setRecommendations] = useState([]);

  const [satUrl, setSatUrl] = useState("");
  const [satImg, setSatImg] = useState(null);
  const satCanvasRef = useRef(null);
  const [satStats, setSatStats] = useState(null);
  const [zoneCount, setZoneCount] = useState(3);

  const [videoFile, setVideoFile] = useState(null);
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const modelRef = useRef(null);
  const detectingRef = useRef(false);
  const [detecting, setDetecting] = useState(false);
  const confidenceRef = useRef(0.5);
  const [confidence, setConfidence] = useState(0.5);
  const [detections, setDetections] = useState([]);

  useEffect(()=>{ const root = document.documentElement; if(dark) root.classList.add("dark"); else root.classList.remove("dark"); },[dark]);

  const onCsvUpload = async (file) => { const data = await parseCsv(file); setRows(data); const cd = toChartData(data).map((d,i)=>({ idx:i, ...d })); setChartData(cd); updateRecommendations(cd, forecast); };

  const trainForecast = async () => {
    if(!chartData || chartData.length === 0) return;
    setModelStatus("Training...");
    const ys = chartData.map(d=>d.yield).filter(v=>Number.isFinite(v));
    if(ys.length < 3){ setModelStatus("Not enough yield data"); return; }
    const xs = ys.map((_,i)=>i);
    const xT = tf.tensor(xs).reshape([xs.length,1]);
    const yT = tf.tensor(ys).reshape([ys.length,1]);
    const m = tf.sequential();
    m.add(tf.layers.dense({units:8, activation:'relu', inputShape:[1]}));
    m.add(tf.layers.dense({units:1}));
    m.compile({optimizer: tf.train.adam(0.05), loss: 'meanSquaredError'});
    try{ await m.fit(xT, yT, {epochs: 100, verbose: 0}); } catch(e){ setModelStatus('Training failed'); return; }
    const futureIdx = [ys.length, ys.length+1, ys.length+2];
    const pred = m.predict(tf.tensor(futureIdx).reshape([futureIdx.length,1]));
    const predVals = Array.from(pred.dataSync()).map(v=>Number(v.toFixed(2)));
    setForecast(futureIdx.map((i,ix)=>({ idx:i, label:`T+${ix+1}`, predYield: predVals[ix] })));
    setModelStatus("Model ready");
    updateRecommendations(chartData, futureIdx.map((i,ix)=>({ idx:i, predYield: predVals[ix] })));
  };

  function updateRecommendations(cd, fc){
    if(!cd || !cd.length){ setRecommendations([]); return; }
    const yields = cd.map(d=>d.yield).filter(x=>Number.isFinite(x));
    const temps = cd.map(d=>d.temp).filter(x=>Number.isFinite(x));
    const rains = cd.map(d=>d.rainfall).filter(x=>Number.isFinite(x));
    const maYield = simpleMovingAverage(yields, 3);
    const trendUp = maYield.length>2 && Number.isFinite(maYield[maYield.length-1]) && Number.isFinite(maYield[Math.max(0,maYield.length-3)]) && (maYield[maYield.length-1] > maYield[Math.max(0,maYield.length-3)]);
    const rainLow = rains.length && Number.isFinite(rains[rains.length-1]) && (rains[rains.length-1] < (rains.reduce((a,b)=>a+b,0)/rains.length)*0.85);
    const heatStress = temps.length && Number.isFinite(temps[temps.length-1]) && (temps[temps.length-1] > 32);
    const futureDip = (fc||[]).some(f=>f.predYield && yields.length && f.predYield < (yields[yields.length-1] * 0.95));
    const recs = [];
    if(rainLow) recs.push({icon: <Droplets className="w-4 h-4"/>, title:"Irrigation watch", detail:"Recent rainfall below typical levels. Consider supplemental irrigation over the next 7–10 days."});
    if(heatStress) recs.push({icon: <Sun className="w-4 h-4"/>, title:"Heat stress risk", detail:"Recent temperatures exceed 32°C. Adjust irrigation timing to early morning/evening and monitor canopy temperature."});
    if(trendUp) recs.push({icon: <Leaf className="w-4 h-4"/>, title:"Yield trend improving", detail:"3-period moving average is rising. Stay the course on nutrient plan; minor N top-up may boost late growth."});
    if(futureDip) recs.push({icon: <LineChartIcon className="w-4 h-4"/>, title:"Possible yield dip ahead", detail:"Short-term forecast suggests softening yields. Re-check pest pressure and soil moisture."});
    setRecommendations(recs);
  }

  const drawGreenness = async (img) => {
    if(!img || !satCanvasRef.current) return;
    const canvas = satCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if(!ctx) return;
    const naturalW = img.naturalWidth || img.width || 512;
    const naturalH = img.naturalHeight || img.height || 512;
    const w = Math.min(1024, naturalW);
    const h = Math.round(naturalH * (w / naturalW));
    canvas.width = w; canvas.height = h;
    ctx.drawImage(img,0,0,w,h);
    const imageData = ctx.getImageData(0,0,w,h);
    const d = imageData.data;
    let min=Infinity, max=-Infinity;
    const greenness = new Float32Array(w*h);
    for(let i=0, j=0;i<d.length;i+=4, j++){
      const r=d[i], g=d[i+1];
      const gIdx = g - r;
      greenness[j]=gIdx; if(gIdx<min)min=gIdx; if(gIdx>max)max=gIdx;
    }
    for(let i=0, j=0;i<d.length;i+=4, j++){
      const v = scaleTo255(greenness[j], min, max);
      d[i] = 20; d[i+1] = v; d[i+2] = 40; d[i+3] = 255;
    }
    ctx.putImageData(imageData,0,0);
    const zones = kMeans1D(Array.from(greenness), zoneCount);
    setSatStats({ min: Number.isFinite(min)?min:0, max: Number.isFinite(max)?max:0, zones });
  };

  function kMeans1D(values, k){
    if(!values || values.length===0) return [];
    k = Math.max(1, Math.floor(k) || 1);
    const minV = Math.min(...values.map(v=>Number.isFinite(v)?v:0));
    const maxV = Math.max(...values.map(v=>Number.isFinite(v)?v:0));
    let cents = Array.from({length:k}, (_,i)=> minV + (k===1?0:(i*(maxV-minV)/(k-1))));
    const iters = 8;
    let assign = new Array(values.length).fill(0);
    for(let t=0;t<iters;t++){
      for(let i=0;i<values.length;i++){ let best=0, bd=Infinity; for(let c=0;c<cents.length;c++){ const d=Math.abs(values[i]-cents[c]); if(d<bd){bd=d; best=c;} } assign[i]=best; }
      for(let c=0;c<cents.length;c++){ const pts=[]; for(let i=0;i<values.length;i++){ if(assign[i]===c) pts.push(values[i]); } if(pts.length) cents[c]=pts.reduce((a,b)=>a+b,0)/pts.length; }
    }
    return cents.map((cent, idx)=>({ zone: idx+1, centroid: cent }));
  }

  const onSatFile = (file) => { if(!file) return; const url = URL.createObjectURL(file); const img = new Image(); img.onload = ()=>{ setSatImg(img); drawGreenness(img); URL.revokeObjectURL(url); }; img.src = url; };
  const onSatUrlLoad = () => { if(!satUrl) return; const img = new Image(); img.crossOrigin = "anonymous"; img.onload = ()=>{ setSatImg(img); drawGreenness(img); }; img.onerror = ()=>{}; img.src = satUrl; };

  useEffect(()=>{ (async()=>{ try{ modelRef.current = await cocoSsd.load({base:'lite_mobilenet_v2'}); }catch(e){ console.warn(e); } })(); },[]);

  useEffect(()=>{ detectingRef.current = detecting; },[detecting]);
  useEffect(()=>{ confidenceRef.current = confidence; },[confidence]);

  const runDetection = async () => {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    if(!video || !canvas || !modelRef.current) return;
    canvas.width = video.videoWidth || video.clientWidth || 640;
    canvas.height = video.videoHeight || video.clientHeight || 360;
    const ctx = canvas.getContext('2d'); if(!ctx) return;
    detectingRef.current = true;
    setDetecting(true);
    const loop = async ()=>{
      if(!detectingRef.current){ return; }
      try{
        ctx.clearRect(0,0,canvas.width,canvas.height);
        const preds = await modelRef.current.detect(video);
        const filtered = preds.filter(p=>p.score >= (confidenceRef.current || 0.5));
        setDetections(filtered.map(p=>({ class:p.class, score: Number(p.score) })));
        filtered.forEach(p=>{
          const [x,y,w,h] = p.bbox;
          ctx.lineWidth = 2; ctx.strokeStyle = '#00ff88'; ctx.fillStyle = 'rgba(0,255,136,0.15)';
          ctx.strokeRect(x,y,w,h); ctx.fillRect(x,y,w,h);
          ctx.font = '14px sans-serif'; ctx.fillStyle = '#111'; ctx.fillText(`${p.class} ${Math.round(p.score*100)}%`, x+4, y+16);
        });
      } catch(e){ console.warn(e); }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  };

  const toggleDetect = () => {
    if(detectingRef.current){ detectingRef.current = false; setDetecting(false); }
    else { detectingRef.current = true; setDetecting(true); runDetection(); }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-background to-muted/30 text-foreground">
      <header className="sticky top-0 z-30 backdrop-blur supports-[backdrop-filter]:bg-background/70 border-b">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-2xl bg-emerald-500/15 flex items-center justify-center"><Leaf className="w-5 h-5 text-emerald-500"/></div>
            <div>
              <h1 className="text-xl font-semibold leading-tight">AgriMind</h1>
              <p className="text-sm text-muted-foreground -mt-1">AI decisions for smarter, resilient farms</p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Sun className="w-4 h-4"/>
              <Switch checked={dark} onCheckedChange={setDark}/>
              <CloudRain className="w-4 h-4"/>
            </div>
            <Button variant="outline" className="gap-2"><Settings className="w-4 h-4"/> Settings</Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 grid lg:grid-cols-12 gap-6">
        <section className="lg:col-span-8 space-y-6">
          <Card className="border-emerald-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2"><Upload className="w-5 h-5"/> Historical Data</CardTitle>
              <CardDescription>Upload CSV with columns like <code>date,yield,rainfall,temp,ndvi</code>. We'll chart trends and create a basic forecast.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-3">
                <Input type="file" accept=".csv" onChange={(e)=>{ if(e.target.files?.[0]) onCsvUpload(e.target.files[0]); }}/>
                <Button onClick={trainForecast} disabled={!chartData.length} className="gap-2"><Wand2 className="w-4 h-4"/> Train Forecast</Button>
                <Badge variant="secondary">{modelStatus}</Badge>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-base">Yield Over Time</CardTitle></CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{top:10,right:20,left:0,bottom:10}}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" hide={false} angle={-10} height={40}/>
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="yield" strokeWidth={2} dot={false} name="Yield" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-base">Rainfall & Temperature</CardTitle></CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{top:10,right:20,left:0,bottom:10}}>
                          <defs>
                            <linearGradient id="rain" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="currentColor" stopOpacity={0.6}/>
                              <stop offset="95%" stopColor="currentColor" stopOpacity={0.05}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" angle={-10} height={40}/>
                          <YAxis />
                          <Tooltip />
                          <Area type="monotone" dataKey="rainfall" fillOpacity={1} fill="url(#rain)" name="Rainfall" />
                          <Line type="monotone" dataKey="temp" strokeWidth={2} dot={false} name="Temp" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {forecast?.length ? (
                <div className="grid md:grid-cols-3 gap-3">
                  {forecast.map((f, i)=> (
                    <Card key={i} className="bg-emerald-500/5 border-emerald-500/30">
                      <CardHeader className="py-3"><CardTitle className="text-sm">Forecast {f.label}</CardTitle></CardHeader>
                      <CardContent className="text-2xl font-semibold">{f.predYield}</CardContent>
                    </Card>
                  ))}
                </div>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2"><MapPinned className="w-5 h-5"/> Satellite Imagery (RGB)</CardTitle>
              <CardDescription>Upload an RGB satellite image (or paste a URL). We'll derive a quick greenness heatmap and simple field zoning.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-col md:flex-row gap-3">
                <Input type="file" accept="image/*" onChange={(e)=>{ if(e.target.files?.[0]) onSatFile(e.target.files[0]); }}/>
                <div className="flex items-center gap-2 w-full">
                  <Input placeholder="https://.../field.png" value={satUrl} onChange={(e)=>setSatUrl(e.target.value)} />
                  <Button variant="outline" onClick={onSatUrlLoad}>Load URL</Button>
                </div>
                <div className="flex items-center gap-2">
                  <Label htmlFor="zones">Zones:</Label>
                  <Select value={String(zoneCount)} onValueChange={(v)=>{ const n=Number(v); setZoneCount(n); if(satImg) drawGreenness(satImg); }}>
                    <SelectTrigger id="zones" className="w-24"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {[2,3,4,5,6].map(n=>(<SelectItem key={n} value={String(n)}>{n}</SelectItem>))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid md:grid-cols-5 gap-4 items-start">
                <div className="md:col-span-3">
                  <div className="border rounded-2xl overflow-hidden">
                    <canvas ref={satCanvasRef} className="w-full h-auto block"/>
                  </div>
                </div>
                <div className="md:col-span-2 space-y-3">
                  <h4 className="font-medium">Greenness Summary</h4>
                  {satStats ? (
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between"><span>Min index</span><span>{satStats.min.toFixed(2)}</span></div>
                      <div className="flex justify-between"><span>Max index</span><span>{satStats.max.toFixed(2)}</span></div>
                      <div className="pt-2">
                        <h5 className="font-medium mb-1">Zones</h5>
                        <div className="flex flex-wrap gap-2">
                          {satStats.zones.map((z)=> (
                            <Badge key={z.zone} variant="secondary">Zone {z.zone}: {z.centroid.toFixed(1)}</Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : <p className="text-muted-foreground text-sm">Load an image to see stats.</p>}
                  <div className="rounded-xl bg-muted p-3 text-xs text-muted-foreground">
                    Tip: For true NDVI and cloud-masking, connect to Google Earth Engine or a satellite API
                    (e.g., Sentinel-2). This demo uses a fast RGB proxy.
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2"><Video className="w-5 h-5"/> Drone Video Analysis</CardTitle>
              <CardDescription>Upload an MP4/MOV drone clip. We'll detect tractors, people, animals, etc. on-device using COCO-SSD.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-col md:flex-row items-center gap-3">
                <Input type="file" accept="video/*" onChange={(e)=>{ if(e.target.files?.[0]){ setVideoFile(URL.createObjectURL(e.target.files[0])); } }} />
                <div className="flex items-center gap-3">
                  <Label className="text-sm">Confidence</Label>
                  <Slider className="w-40" min={0.3} max={0.9} step={0.05} value={[confidence]} onValueChange={(v)=>{ setConfidence(v[0]); confidenceRef.current = v[0]; }} />
                  <Badge variant="outline">{Math.round(confidence*100)}%</Badge>
                </div>
                <Button onClick={toggleDetect} disabled={!videoFile} className="gap-2">
                  {detecting ? <Pause className="w-4 h-4"/> : <Play className="w-4 h-4"/>}
                  {detecting? 'Pause Detection' : 'Start Detection'}
                </Button>
              </div>
              <div className="grid md:grid-cols-5 gap-4 items-start">
                <div className="md:col-span-3 relative rounded-2xl overflow-hidden border">
                  <video ref={videoRef} src={videoFile || undefined} controls className="w-full h-auto block" onPlay={()=>{ if(detectingRef.current) runDetection(); }} />
                  <canvas ref={overlayRef} className="absolute inset-0 pointer-events-none" style={{top:0,left:0,width:'100%',height:'100%'}}/>
                </div>
                <div className="md:col-span-2">
                  <h4 className="font-medium mb-2">Detections</h4>
                  <div className="h-56 overflow-auto rounded-xl border p-2 text-sm bg-background/50">
                    {detections.length? detections.map((d,i)=>(
                      <div key={i} className="flex items-center justify-between py-1 border-b last:border-none">
                        <span className="capitalize">{d.class}</span>
                        <Badge variant="secondary">{Math.round(d.score*100)}%</Badge>
                      </div>
                    )): <p className="text-muted-foreground">No detections yet.</p>}
                  </div>
                  <div className="rounded-xl bg-muted p-3 text-xs text-muted-foreground mt-3">
                    For longer videos, process on a server and summarize key events (equipment idle time,
                    livestock count, safety alerts).
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <aside className="lg:col-span-4 space-y-6">
          <Card className="border-emerald-500/20">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2"><LineChartIcon className="w-5 h-5"/> Smart Recommendations</CardTitle>
              <CardDescription>Data- and vision-driven suggestions for the next 2 weeks.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {recommendations.length ? recommendations.map((r, i)=>(
                <div key={i} className="p-3 rounded-xl bg-emerald-500/5 border border-emerald-500/20">
                  <div className="flex items-center gap-2 font-medium">{r.icon} {r.title}</div>
                  <p className="text-sm text-muted-foreground mt-1">{r.detail}</p>
                </div>
              )) : (
                <p className="text-sm text-muted-foreground">Upload data to see tailored recommendations.</p>
              )}
              <div className="pt-1">
                <h4 className="font-medium mb-2">Operational Tips</h4>
                <ul className="text-sm list-disc pl-5 space-y-1 text-muted-foreground">
                  <li>Calibrate sensors after rainfall events to reduce noise in yield mapping.</li>
                  <li>Use 3+ years of history for stable forecasts; include soil tests if available.</li>
                  <li>Segment fields into low/med/high vigor zones for variable-rate inputs.</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2"><ImageIcon className="w-5 h-5"/> Google Earth Integration</CardTitle>
              <CardDescription>Connect your imagery provider.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <p>To pull tiles directly from Google Earth/Maps or Earth Engine, add your API key and set up a small proxy on your backend to sign requests. This prototype accepts direct image URLs or uploads.</p>
              <div className="rounded-xl bg-muted p-3">
                <div className="font-medium mb-1">Backend TODOs</div>
                <ul className="list-disc pl-5 space-y-1">
                  <li>OAuth/API-key vault + rate limiting</li>
                  <li>Earth Engine tasks (NDVI, EVI, cloud mask) with exports to tiled PNG</li>
                  <li>Job queue + caching for repeat fields</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2"><Tractor className="w-5 h-5"/> Field Summary</CardTitle>
              <CardDescription>At-a-glance health & risk.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl border bg-background/50">
                  <div className="text-xs text-muted-foreground">Greenness Variability</div>
                  <div className="text-2xl font-semibold">{satStats? Math.abs(satStats.max - satStats.min).toFixed(1) : '–'}</div>
                </div>
                <div className="p-3 rounded-xl border bg-background/50">
                  <div className="text-xs text-muted-foreground">Zones</div>
                  <div className="text-2xl font-semibold">{satStats? zoneCount : '–'}</div>
                </div>
                <div className="p-3 rounded-xl border bg-background/50">
                  <div className="text-xs text-muted-foreground">Latest Yield</div>
                  <div className="text-2xl font-semibold">{chartData?.length? (chartData[chartData.length-1]?.yield ?? '–') : '–'}</div>
                </div>
                <div className="p-3 rounded-xl border bg-background/50">
                  <div className="text-xs text-muted-foreground">Forecast (T+1)</div>
                  <div className="text-2xl font-semibold">{forecast?.[0]?.predYield ?? '–'}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </aside>
      </main>

      <footer className="border-t py-6 text-center text-xs text-muted-foreground">Prototype – For demonstration only. Not for operational decisions without expert review.</footer>
    </div>
  );
}
