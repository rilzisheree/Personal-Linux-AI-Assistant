export type Role = 'user' | 'assistant' | string;

export type Message = {
  role: Role;
  content: string;
};

export type ConversationSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type Conversation = ConversationSummary & {
  messages: Message[];
};

export function withMessages(conversation: Conversation): Conversation {
  return { ...conversation, messages: Array.isArray(conversation.messages) ? conversation.messages : [] };
}

export type HealthResponse = {
  ok: boolean;
  service: string;
  ollama_url: string;
};

export class LuraApiError extends Error {
  status: number;
  code: 'unauthorized' | 'unavailable' | 'bad-response';

  constructor(message: string, status = 0, code: LuraApiError['code'] = 'bad-response') {
    super(message);
    this.name = 'LuraApiError';
    this.status = status;
    this.code = code;
  }
}

const SESSION_TOKEN_KEY = 'lura_session_token';
const configuredApiBaseUrl = (import.meta.env.VITE_LURA_API_URL || '').trim().replace(/\/+$/, '');

function apiUrl(path: string): string {
  return `${configuredApiBaseUrl}${path}`;
}

function getSessionToken(): string | null {
  try {
    return window.sessionStorage.getItem(SESSION_TOKEN_KEY);
  } catch {
    return null;
  }
}

function setSessionToken(token: string): void {
  try {
    window.sessionStorage.setItem(SESSION_TOKEN_KEY, token);
  } catch {
    // The HttpOnly cookie remains available when session storage is blocked.
  }
}

function clearSessionToken(): void {
  try {
    window.sessionStorage.removeItem(SESSION_TOKEN_KEY);
  } catch {
    // There is nothing else to clear when session storage is blocked.
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  const headers = new Headers(init?.headers);
  headers.set('Accept', 'application/json');
  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const sessionToken = getSessionToken();
  if (sessionToken && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${sessionToken}`);
  }
  try {
    response = await fetch(apiUrl(path), {
      credentials: 'include',
      ...init,
      headers,
    });
  } catch {
    throw new LuraApiError(
      'The local API could not be reached. Is the Lura service running?',
      0,
      'unavailable',
    );
  }

  if (response.status === 401 || response.status === 403) {
    clearSessionToken();
    throw new LuraApiError('Your Lura session has expired. Please sign in again.', response.status, 'unauthorized');
  }
  if (!response.ok) {
    let detail = '';
    try {
      const body = (await response.json()) as { detail?: string; message?: string };
      detail = body.detail || body.message || '';
    } catch {
      // The service may return an empty or non-JSON error body.
    }
    throw new LuraApiError(detail || `Lura returned an error (${response.status}).`, response.status);
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new LuraApiError('Lura returned an unreadable response.', response.status);
  }
}

export function getHealth() {
  return request<HealthResponse>('/api/health');
}

export async function login(password: string) {
  const result = await request<{ authenticated: true; session_token?: string }>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ password }),
  });
  if (result.session_token) setSessionToken(result.session_token);
  return result;
}

export async function logout() {
  try {
    return await request<{ ok: true }>('/api/auth/logout', { method: 'POST' });
  } finally {
    clearSessionToken();
  }
}

export function getMe() {
  return request<{ user: { id: string; authenticated: boolean } }>('/api/me');
}

export function getModels() {
  return request<{ models: string[] }>('/api/models');
}

export function getConversations() {
  return request<{ conversations: ConversationSummary[] }>('/api/conversations');
}

export function getConversation(id: string) {
  return request<{ conversation: Conversation }>(`/api/conversations/${encodeURIComponent(id)}`);
}

export function createConversation(title?: string) {
  return request<{ conversation: Conversation }>('/api/conversations', {
    method: 'POST',
    body: JSON.stringify(title ? { title } : {}),
  });
}

type StreamEvent =
  | { type: 'started'; data: { conversation_id?: string } }
  | { type: 'token'; data: { content?: string } }
  | { type: 'done'; data: { conversation_id?: string; message?: string } }
  | { type: 'error'; data: { message?: string } };

export async function streamMessage(
  id: string,
  payload: { content: string; model?: string },
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
) {
  let response: Response;
  const headers = new Headers({
    Accept: 'text/event-stream',
    'Content-Type': 'application/json',
  });
  const sessionToken = getSessionToken();
  if (sessionToken) headers.set('Authorization', `Bearer ${sessionToken}`);
  try {
    response = await fetch(apiUrl(`/api/conversations/${encodeURIComponent(id)}/messages`), {
      method: 'POST',
      credentials: 'include',
      headers,
      body: JSON.stringify(payload),
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new LuraApiError('The local API stopped responding while sending your message.', 0, 'unavailable');
  }

  if (response.status === 401 || response.status === 403) {
    clearSessionToken();
    throw new LuraApiError('Your Lura session has expired. Please sign in again.', response.status, 'unauthorized');
  }
  if (!response.ok || !response.body) {
    throw new LuraApiError(`Lura could not start this response (${response.status}).`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const processBlock = (block: string) => {
    let eventName = 'message';
    const dataLines: string[] = [];
    block.split(/\r?\n/).forEach((line) => {
      if (line.startsWith('event:')) eventName = line.slice(6).trim();
      if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    });
    if (!dataLines.length) return;
    try {
      const data = JSON.parse(dataLines.join('\n')) as StreamEvent['data'];
      if (eventName === 'started' || eventName === 'token' || eventName === 'done' || eventName === 'error') {
        onEvent({ type: eventName, data } as StreamEvent);
      }
    } catch {
      onEvent({ type: 'error', data: { message: 'Lura sent an invalid stream event.' } });
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || '';
    blocks.forEach(processBlock);
    if (done) break;
  }
  if (buffer.trim()) processBlock(buffer);
}