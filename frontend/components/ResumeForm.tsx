"use client";
import { useState } from "react";

export default function ResumeForm() {
  const [resume, setResume] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);

  const submitText = async () => {
    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: resume }),
    });
    setResult(await res.json());
  };

  const submitPDF = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/predict-pdf", {
      method: "POST",
      body: formData,
    });

    setResult(await res.json());
  };

  return (
    <div className="max-w-xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">TalentLens Resume Analyzer</h1>

      <textarea
        placeholder="Paste resume text"
        className="w-full border p-2 mb-2"
        rows={6}
        value={resume}
        onChange={(e) => setResume(e.target.value)}
      />

      <button
        onClick={submitText}
        className="bg-black text-white px-4 py-2 mr-2"
      >
        Analyze Text
      </button>

      <hr className="my-4" />

      <input
        type="file"
        accept="application/pdf"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />

      <button
        onClick={submitPDF}
        className="bg-blue-600 text-white px-4 py-2 ml-2"
      >
        Upload PDF
      </button>

      {result && (
        <div className="mt-6 border p-4">
          <p><b>Category:</b> {result.category}</p>
          <p><b>Confidence:</b> {result.confidence}%</p>
          <p><b>Top Keywords:</b></p>
          <ul className="list-disc ml-6">
            {result.keywords.map((k: string, i: number) => (
              <li key={i}>{k}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
