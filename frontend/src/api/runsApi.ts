import { API_BASE_URL, API_ENDPOINTS } from '../config';
import type { RunRecord, RunsReport } from '../types';

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const message = payload?.error?.message || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function startRun(keywords: string[], n: number): Promise<RunRecord> {
  const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.RUNS_START}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ keywords, n }),
  });
  const payload = await parseResponse<{ run: RunRecord }>(response);
  return payload.run;
}

export async function fetchRuns(limit = 50): Promise<RunRecord[]> {
  const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.RUNS_LIST}?limit=${limit}`);
  const payload = await parseResponse<{ runs: RunRecord[] }>(response);
  return payload.runs;
}

export async function fetchRunDetail(runId: string, includeIdeas = false): Promise<RunRecord> {
  const response = await fetch(
    `${API_BASE_URL}${API_ENDPOINTS.RUNS_DETAIL}/${runId}?includeIdeas=${includeIdeas}`
  );
  const payload = await parseResponse<{ run: RunRecord }>(response);
  return payload.run;
}

export async function fetchRunsReport(): Promise<RunsReport> {
  const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.RUNS_REPORT}`);
  return parseResponse<RunsReport>(response);
}
