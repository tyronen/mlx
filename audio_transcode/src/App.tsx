import {useEffect, useRef, useState} from 'react'
import './App.css'

const SERVER = "gb1maa23kmlnns-8888.proxy.runpod.net";

type Segment = {
    id?: string
    chunk_id?: number
    start_sec: number
    end_sec: number
    text: string
}

export default function App() {
    const [isRecording, setIsRecording] = useState(false)
    const [segments, setSegments] = useState<Segment[]>([])
    const [recordedPCM, setRecordedPCM] = useState<Int16Array[]>([])
    const [editableTranscript, setEditableTranscript] = useState('')
    const [elapsed, setElapsed] = useState(0)
    const [isProcessing, setIsProcessing] = useState(false)
    const [selectedModel, setSelectedModel] = useState<string>("openai/whisper-large-v3");
    const [trainResults, setTrainResults] = useState<{ summary: string; pre_loss: number; post_loss: number }[]>([]);
    const [isTraining, setIsTraining] = useState(false);

    const handleModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
        const model = e.target.value;
        setSelectedModel(model);
        const form = new FormData();
        form.append("model_name", model);
        await fetch(`https://${SERVER}/model`, {
            method: "POST",
            body: form,
        });
    };

    const audioCtxRef = useRef<AudioContext | null>(null)
    const workletNodeRef = useRef<AudioWorkletNode | null>(null)
    const wsRef = useRef<WebSocket | null>(null)
    const prevWindowEndRef = useRef<number>(0);
    const audioBufferRef = useRef<Int16Array[]>([]);
    const timerRef = useRef<number | null>(null)
    const transcriptRef = useRef<HTMLTextAreaElement | null>(null)

    // Start/stop timer
    useEffect(() => {
        if (isRecording) {
            setElapsed(0)
            timerRef.current = window.setInterval(() => setElapsed(e => e + 1), 1000)
            // setTrainResult(null);
        } else {
            if (timerRef.current) clearInterval(timerRef.current)
            timerRef.current = null
        }
    }, [isRecording])

    // Live transcription & PCM buffering
    useEffect(() => {
        if (!isRecording) return

        setSegments([])
        setRecordedPCM([])

        async function startRecording() {
            if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
                wsRef.current = new WebSocket(`wss://${SERVER}/ws`)
                wsRef.current.onmessage = e => {
                    try {
                        // 1) Parse the server’s JSON payload
                        const {chunk_id, tokens, word_timestamps, start_sec, duration} = JSON.parse(e.data);
                        const windowStart = start_sec;
                        const windowEnd = start_sec + duration;

                        // 2) Build the new segments for this window
                        const newSegs: Segment[] = tokens.map((tok: string, i: number) => {
                            const [ws, we] = word_timestamps[i];
                            return {
                                id: `${chunk_id}-${i}`,
                                start_sec: windowStart + ws,
                                end_sec: windowStart + we,
                                text: tok,
                            };
                        });

                        // 3) Compute removal range:
                        //    any old words in [windowStart, removalEnd) must go away.
                        const removalEnd = Math.max(prevWindowEndRef.current, windowEnd);

                        // 4) Update segments state:
                        setSegments(prev => {
                            // keep anything completely before the window or completely after removalEnd
                            const filtered = prev.filter(seg =>
                                seg.end_sec <= windowStart ||
                                seg.start_sec >= removalEnd
                            );
                            // merge in new ones and sort by time
                            return [...filtered, ...newSegs].sort((a, b) =>
                                a.start_sec - b.start_sec
                            );
                        });

                        // 5) Remember the new furthest end
                        prevWindowEndRef.current = removalEnd;

                    } catch (err) {
                        console.error("bad ws message", err);
                    }
                };

                wsRef.current.onopen = () => {
                    // flush any buffered audio once connection opens
                    audioBufferRef.current.forEach(arr => wsRef.current!.send(arr.buffer));
                    audioBufferRef.current = [];
                };
                wsRef.current.onclose = () => setIsProcessing(false)
            }
            const ws = wsRef.current!

            const stream = await navigator.mediaDevices.getUserMedia({audio: true})
            const ctx = new AudioContext({sampleRate: 16000})
            audioCtxRef.current = ctx
            await ctx.audioWorklet.addModule('/pcm-processor.js')
            const source = ctx.createMediaStreamSource(stream)
            const worklet = new AudioWorkletNode(ctx, 'pcm-processor')
            workletNodeRef.current = worklet

            worklet.port.onmessage = (e) => {
                const float32 = e.data as Float32Array
                const int16 = new Int16Array(float32.length)
                for (let i = 0; i < float32.length; i++) {
                    const s = Math.max(-1, Math.min(1, float32[i]))
                    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
                }
                setRecordedPCM(prev => [...prev, int16]);
                // buffer the PCM chunk
                audioBufferRef.current.push(int16);
                // if socket open, flush buffer
                if (ws.readyState === WebSocket.OPEN) {
                    audioBufferRef.current.forEach(buf => ws.send(buf.buffer));
                    audioBufferRef.current = [];
                }
            }


            source.connect(worklet).connect(ctx.destination)
        }

        startRecording()

        return () => {
            audioCtxRef.current?.close()
            audioCtxRef.current = null
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(new ArrayBuffer(0))
                setIsProcessing(true)
            }
            workletNodeRef.current = null
        }
    }, [isRecording])

    // Build editable transcript
    useEffect(() => {
        setEditableTranscript(segments.map(s => s.text).join(""));
    }, [segments])

    // Auto-scroll
    useEffect(() => {
        const el = transcriptRef.current
        if (el) el.scrollTo({top: el.scrollHeight, behavior: 'smooth'})
    }, [segments])

    // Train handler
    const handleTrain = async () => {
        setIsTraining(true);
        // setTrainResult(null);
        const total = recordedPCM.reduce((sum, a) => sum + a.length, 0)
        const allPCM = new Int16Array(total)
        let off = 0
        recordedPCM.forEach(arr => {
            allPCM.set(arr, off);
            off += arr.length
        })
        const blob = new Blob([allPCM.buffer], {type: 'audio/pcm'})
        const form = new FormData()
        form.append('audio', blob, 'audio.pcm')
        form.append('transcript', editableTranscript)

        const res = await fetch(`https://${SERVER}/train`, {method: 'POST', body: form});
        const data = await res.json();
        if (res.ok) {
            console.log(data);
            setTrainResults(prev => [{
                summary: data.summary,
                pre_loss: data.pre_loss,
                post_loss: data.post_loss
            }, ...prev]);
        } else {
            // setTrainResult(null);
            console.error('Train failed:', data);
        }
        setIsTraining(false);
    }

    return (
        <div className="app">
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                <h1>Voice Transcriber Demo</h1>
            </div>
            <div className="controls">
                {/* Left side: Record/Train → timer & processing */}
                <div className="left-group">
                    <div className="buttons-row">
                        <button
                            onClick={() => setIsRecording(r => !r)}
                            className="record"
                        >
        <span
            style={{
                borderRadius: isRecording ? 0 : '50%',
            }}
        />
                            {isRecording ? 'Stop' : 'Record'}
                        </button>

                        <button
                            disabled={isRecording || segments.length === 0 || isTraining}
                            onClick={handleTrain}
                            className="train"
                        >
                            {isTraining && <span className="spinner"/>}
                            {isTraining ? 'Training...' : 'Train'}
                        </button>
                    </div>

                    <div className="status-row">
                        <div className="stopwatch">⏱ {elapsed}s</div>
                        {!isRecording && isProcessing && (
                            <div className="processing">Receiving more text…</div>
                        )}
                    </div>
                </div>

                {/* Right side: Model selector */}
                <div className="right-group" style={{display: 'flex', alignItems: 'center'}}>
                    <label htmlFor="model-select">
                        Model:
                    </label>
                    <select
                        id="model-select"
                        value={selectedModel}
                        onChange={handleModelChange}
                    >
                        <option value="openai/whisper-tiny.en">Tiny English</option>
                        <option value="openai/whisper-tiny">Tiny</option>
                        <option value="openai/whisper-base.en">Base English</option>
                        <option value="openai/whisper-base">Base</option>
                        <option value="distil-whisper/distil-small.en">Distil Small</option>
                        <option value="distil-whisper/distil-medium.en">Distil Medium</option>
                        <option value="openai/whisper-small">Small</option>
                        <option value="distil-whisper/distil-large-v3">Distil Large v3</option>
                        <option value="openai/whisper-large-v3-turbo">Turbo Large v3</option>
                        <option value="openai/whisper-medium">Medium</option>
                        <option value="openai/whisper-large-v3">Large v3</option>
                    </select>
                </div>
            </div>
            <textarea
                ref={transcriptRef}
                value={editableTranscript}
                readOnly={isRecording}
                onChange={(e) => setEditableTranscript(e.target.value)}
                rows={10}
            />
            {trainResults.length > 0 && (
                <table style={{marginTop: 12, width: '100%', borderCollapse: 'collapse'}}>
                    <thead>
                    <tr>
                        <th style={{textAlign: 'left', borderBottom: '1px solid #444', padding: '4px'}}>Summary</th>
                        <th style={{textAlign: 'right', borderBottom: '1px solid #444', padding: '4px'}}>Pre-loss</th>
                        <th style={{textAlign: 'right', borderBottom: '1px solid #444', padding: '4px'}}>Post-loss</th>
                    </tr>
                    </thead>
                    <tbody>
                    {trainResults.map((r, idx) => (
                        <tr key={idx}>
                            <td style={{padding: '4px', borderBottom: '1px solid #333'}}>{r.summary}</td>
                            <td style={{padding: '4px', borderBottom: '1px solid #333', textAlign: 'right'}}>
                                {r.pre_loss.toFixed(3)}
                            </td>
                            <td style={{padding: '4px', borderBottom: '1px solid #333', textAlign: 'right'}}>
                                {r.post_loss.toFixed(3)}
                            </td>
                        </tr>
                    ))}
                    </tbody>
                </table>
            )}
        </div>
    )
}