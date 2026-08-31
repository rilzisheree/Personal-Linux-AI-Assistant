import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { Menu, Plus, ArrowUp, ChevronDown, LogOut, RefreshCw, Server, Square, Wifi, WifiOff, X } from 'lucide-react';
import { Route, Switch, Router as WouterRouter, useLocation } from 'wouter';
import { LuraApiError, createConversation, getConversation, getConversations, getHealth, getMe, getModels, login, logout, streamMessage, withMessages, type Conversation, type ConversationSummary, type HealthResponse, type Message } from '@/lib/lura-api';

function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-3" data-testid="brand-lura">
      <div className="relative flex size-9 shrink-0 items-center justify-center rounded-[11px] bg-[hsl(var(--sidebar-primary))] text-[hsl(var(--sidebar))]">
        <span className="absolute size-3 rounded-full border-2 border-current" />
        <span className="absolute h-4 w-px rotate-45 bg-current" />
        <span className="absolute h-4 w-px -rotate-45 bg-current" />
      </div>
      {!compact && <div><div className="font-mono text-[13px] font-bold tracking-[0.2em] text-[hsl(var(--sidebar-foreground))]">LURA</div><div className="font-mono text-[9px] uppercase tracking-[0.16em] text-[hsl(var(--sidebar-foreground)/.48)]">local intelligence</div></div>}
    </div>
  );
}

function ConnectionPill({ health, checking = false }: { health: HealthResponse | null; checking?: boolean }) {
  const connected = Boolean(health?.ok);
  return <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.12em] ${connected ? 'border-[hsl(var(--primary)/.22)] bg-[hsl(var(--primary)/.08)] text-[hsl(var(--primary))]' : 'border-[hsl(var(--destructive)/.22)] bg-[hsl(var(--destructive)/.07)] text-[hsl(var(--destructive))]'}`} data-testid="status-connection">{connected ? <Wifi size={12} /> : <WifiOff size={12} />}<span>{checking ? 'checking channel' : connected ? 'local channel live' : 'channel unavailable'}</span><span className={`size-1.5 rounded-full ${connected ? 'bg-[hsl(var(--primary))] lura-pulse' : 'bg-[hsl(var(--destructive))]'}`} /></div>;
}

function ErrorNotice({ message, onRetry, testId = 'status-api-error' }: { message: string; onRetry?: () => void; testId?: string }) {
  return <div className="rounded-xl border border-[hsl(var(--destructive)/.22)] bg-[hsl(var(--destructive)/.06)] p-4 text-sm text-[hsl(var(--destructive))]" data-testid={testId}><p className="leading-6">{message}</p>{onRetry && <button type="button" onClick={onRetry} className="mt-3 inline-flex items-center gap-2 rounded-lg border border-current/20 px-3 py-2 text-xs font-semibold transition-colors hover:bg-[hsl(var(--destructive)/.08)]" data-testid="button-retry-api"><RefreshCw size={13} /> Try again</button>}</div>;
}

function LoginPage() {
  const [, setLocation] = useLocation();
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [channelState, setChannelState] = useState<'checking' | 'live' | 'offline'>('checking');

  useEffect(() => { getHealth().then((health) => setChannelState(health.ok ? 'live' : 'offline')).catch(() => setChannelState('offline')); }, []);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!password.trim()) { setError('Enter the local access password to continue.'); return; }
    setSubmitting(true); setError('');
    try { await login(password); setLocation('/'); } catch (cause) { setError(cause instanceof LuraApiError ? cause.message : 'The password could not be verified.'); } finally { setSubmitting(false); }
  }

  return (
    <main className="grain flex min-h-[100dvh] items-center justify-center overflow-hidden bg-[hsl(var(--sidebar))] px-5 py-10 text-[hsl(var(--sidebar-foreground))]">
      <div className="pointer-events-none absolute left-[-15vw] top-[-18vw] size-[55vw] rounded-full border border-[hsl(var(--sidebar-primary)/.1)]" />
      <div className="pointer-events-none absolute bottom-[-20vw] right-[-15vw] size-[60vw] rounded-full border border-[hsl(var(--accent)/.12)]" />
      <div className="lura-rise relative grid w-full max-w-[940px] overflow-hidden rounded-2xl border border-[hsl(var(--sidebar-border))] bg-[hsl(var(--sidebar)/.72)] shadow-[0_28px_80px_hsl(231_28%_8%/.28)] md:grid-cols-[1.04fr_.96fr]">
        <section className="hidden flex-col justify-between border-r border-[hsl(var(--sidebar-border))] p-10 md:flex"><BrandMark /><div><div className="mb-5 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-[hsl(var(--sidebar-primary))]"><span className="size-1.5 rounded-full bg-[hsl(var(--sidebar-primary))]" /> Private channel</div><h1 className="max-w-sm text-4xl font-semibold leading-[1.08] tracking-[-0.04em]">A quiet window into your local mind.</h1><p className="mt-5 max-w-sm text-sm leading-7 text-[hsl(var(--sidebar-foreground)/.58)]">Lura keeps your conversations close to home. Sign in to open the companion window.</p></div><div className="font-mono text-[10px] uppercase tracking-[.14em] text-[hsl(var(--sidebar-foreground)/.38)]">No cloud relay <span className="mx-2 text-[hsl(var(--sidebar-primary)/.6)]">/</span> Your machine</div></section>
       <section className="bg-[hsl(var(--card)/.96)] px-6 py-8 text-[hsl(var(--foreground))] sm:px-10 sm:py-12"><div className="mb-12 md:hidden"><BrandMark /></div><div className="mb-9"><p className="mb-3 font-mono text-[10px] uppercase tracking-[.18em] text-[hsl(var(--primary))]">Access terminal</p><h2 className="text-3xl font-semibold tracking-[-0.04em]">Welcome back.</h2><p className="mt-3 text-sm leading-6 text-[hsl(var(--muted-foreground))]">This door only opens for the local user.</p></div><form onSubmit={handleLogin} className="space-y-5" data-testid="form-login"><input aria-hidden="true" autoComplete="username" className="sr-only" name="username" tabIndex={-1} type="text" value="local-user" readOnly /><label className="block"><span className="mb-2 block font-mono text-[10px] uppercase tracking-[.14em] text-[hsl(var(--muted-foreground))]">Access password</span><input autoFocus autoComplete="current-password" type="password" value={password} onChange={(event) => { setPassword(event.target.value); if (error) setError(''); }} placeholder="Enter your local password" className="h-12 w-full rounded-xl border border-[hsl(var(--border))] bg-[hsl(var(--background))] px-4 text-sm outline-none transition-colors placeholder:text-[hsl(var(--muted-foreground)/.58)] focus:border-[hsl(var(--primary))] focus:ring-4 focus:ring-[hsl(var(--primary)/.1)]" data-testid="input-password" /></label>{error && <ErrorNotice message={error} testId="status-login-error" />}<button type="submit" disabled={submitting} className="flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-[hsl(var(--primary))] text-sm font-semibold text-[hsl(var(--primary-foreground))] transition-transform hover:-translate-y-0.5 disabled:opacity-60" data-testid="button-login">{submitting ? 'Opening channel…' : 'Open Lura'} <ArrowUp className="rotate-45" size={16} /></button></form><div className="mt-10 flex items-center gap-2 border-t border-[hsl(var(--border))] pt-5 font-mono text-[10px] uppercase tracking-[.1em] text-[hsl(var(--muted-foreground))]" data-testid="status-login-connection"><span className={`size-1.5 rounded-full ${channelState === 'live' ? 'bg-[hsl(var(--primary))]' : channelState === 'offline' ? 'bg-[hsl(var(--destructive))]' : 'bg-[hsl(var(--accent))] lura-pulse'}`} />{channelState === 'live' ? 'API detected on this origin' : channelState === 'offline' ? 'API not detected' : 'Locating local API'}</div></section>
      </div>
    </main>
  );
}

function Sidebar({ conversations, activeId, onSelect, onNew, onLogout, onClose, busy }: { conversations: ConversationSummary[]; activeId: string | null; onSelect: (id: string) => void; onNew: () => void; onLogout: () => void; onClose?: () => void; busy?: boolean }) {
  const formatDate = (value: string) => { const date = new Date(value); if (Number.isNaN(date.getTime())) return 'recently'; const now = new Date(); if (date.toDateString() === now.toDateString()) return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }); return date.toLocaleDateString([], { month: 'short', day: 'numeric' }); };
  return <aside className="flex min-h-[100dvh] w-full max-w-[320px] flex-col bg-[hsl(var(--sidebar))] text-[hsl(var(--sidebar-foreground))] md:max-w-none" data-testid="sidebar-conversations"><div className="flex items-center justify-between border-b border-[hsl(var(--sidebar-border))] px-5 py-5"><BrandMark />{onClose && <button type="button" onClick={onClose} className="rounded-lg p-2 text-[hsl(var(--sidebar-foreground)/.65)] hover:bg-[hsl(var(--sidebar-accent))] md:hidden" aria-label="Close menu" data-testid="button-close-menu"><X size={18} /></button>}</div><div className="p-4"><button type="button" onClick={onNew} disabled={busy} className="flex h-11 w-full items-center justify-between rounded-xl bg-[hsl(var(--sidebar-primary))] px-4 text-sm font-semibold text-[hsl(var(--sidebar-primary-foreground))] transition-transform hover:-translate-y-0.5 disabled:opacity-60" data-testid="button-new-conversation"><span className="flex items-center gap-2"><Plus size={16} /> New conversation</span><span className="font-mono text-[10px] opacity-60">N</span></button></div><div className="flex items-center justify-between px-5 pb-3 pt-2"><span className="font-mono text-[10px] uppercase tracking-[.16em] text-[hsl(var(--sidebar-foreground)/.42)]">History</span><span className="font-mono text-[10px] text-[hsl(var(--sidebar-foreground)/.36)]" data-testid="text-conversation-count">{conversations.length.toString().padStart(2, '0')}</span></div><div className="min-h-0 flex-1 overflow-y-auto px-3 pb-5">{conversations.length === 0 ? <div className="mx-2 mt-4 rounded-xl border border-dashed border-[hsl(var(--sidebar-border))] p-4 text-xs leading-5 text-[hsl(var(--sidebar-foreground)/.45)]" data-testid="empty-conversations">Your recent conversations will appear here.</div> : <div className="space-y-1">{conversations.map((item) => <button type="button" key={item.id} onClick={() => onSelect(item.id)} className={`group w-full rounded-xl px-3 py-3 text-left transition-colors ${activeId === item.id ? 'bg-[hsl(var(--sidebar-accent))] text-[hsl(var(--sidebar-accent-foreground))]' : 'text-[hsl(var(--sidebar-foreground)/.68)] hover:bg-[hsl(var(--sidebar-accent)/.65)] hover:text-[hsl(var(--sidebar-foreground))]'}`} data-testid={`button-conversation-${item.id}`}><div className="truncate text-[13px] font-medium">{item.title || 'Untitled conversation'}</div><div className="mt-1 font-mono text-[9px] uppercase tracking-[.1em] text-[hsl(var(--sidebar-foreground)/.35)]">{formatDate(item.updated_at)}</div></button>)}</div>}</div><div className="border-t border-[hsl(var(--sidebar-border))] p-4"><div className="mb-3 flex items-center gap-2 px-2 font-mono text-[9px] uppercase tracking-[.12em] text-[hsl(var(--sidebar-foreground)/.38)]"><span className="size-1.5 rounded-full bg-[hsl(var(--sidebar-primary))]" /> Single local user</div><button type="button" onClick={onLogout} className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-xs text-[hsl(var(--sidebar-foreground)/.58)] hover:bg-[hsl(var(--sidebar-accent))] hover:text-[hsl(var(--sidebar-foreground))]" data-testid="button-logout"><LogOut size={14} /> Sign out</button></div></aside>;
}

function MessageBubble({ message, index }: { message: Message; index: number }) {
  const isUser = message.role === 'user';
  return <div className={`lura-rise flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`} style={{ animationDelay: `${Math.min(index * 35, 240)}ms` }} data-testid={`message-${isUser ? 'user' : 'assistant'}-${index}`}>{!isUser && <div className="mt-1 flex size-7 shrink-0 items-center justify-center rounded-lg border border-[hsl(var(--primary)/.25)] bg-[hsl(var(--primary)/.08)]"><span className="size-2 rounded-full bg-[hsl(var(--primary))]" /></div>}<div className={`max-w-[min(720px,86%)] ${isUser ? 'rounded-2xl rounded-br-md bg-[hsl(var(--primary))] px-4 py-3 text-[hsl(var(--primary-foreground))]' : 'max-w-[780px] pt-1 text-[hsl(var(--foreground)/.86)]'}`}><div className={`mb-1 font-mono text-[9px] uppercase tracking-[.14em] ${isUser ? 'text-[hsl(var(--primary-foreground)/.6)]' : 'text-[hsl(var(--muted-foreground))]'}`}>{isUser ? 'you' : 'lura'}</div><div className={`message-copy whitespace-pre-wrap text-[14px] leading-7 ${isUser ? '' : 'border-b border-[hsl(var(--border)/.52)] pb-5'}`} data-testid={`text-message-content-${index}`}>{message.content || <span className="lura-pulse inline-block text-[hsl(var(--muted-foreground))]">thinking</span>}</div></div></div>;
}

function EmptyWorkspace({ onNew }: { onNew: () => void }) {
  return <div className="flex min-h-0 flex-1 items-center justify-center px-6 py-12" data-testid="empty-chat"><div className="w-full max-w-lg text-center"><div className="mx-auto mb-7 flex size-16 items-center justify-center rounded-[22px] border border-[hsl(var(--primary)/.22)] bg-[hsl(var(--primary)/.07)]"><div className="size-5 rounded-full border-[3px] border-[hsl(var(--primary))]" /></div><p className="mb-3 font-mono text-[10px] uppercase tracking-[.2em] text-[hsl(var(--primary))]">Channel ready</p><h2 className="text-3xl font-semibold tracking-[-.05em] text-[hsl(var(--foreground))]">What’s on your mind?</h2><p className="mx-auto mt-4 max-w-sm text-sm leading-6 text-[hsl(var(--muted-foreground))]">Start a private thread with the assistant running on your own machine.</p><button type="button" onClick={onNew} className="mt-7 inline-flex items-center gap-2 rounded-xl border border-[hsl(var(--border))] bg-[hsl(var(--card))] px-4 py-3 text-sm font-semibold shadow-[var(--shadow-sm)] transition-transform hover:-translate-y-0.5" data-testid="button-empty-new-conversation"><Plus size={16} /> Start a conversation</button></div></div>;
}

function ChatWorkspace() {
  const [, setLocation] = useLocation();
  const [authState, setAuthState] = useState<'checking' | 'ready' | 'expired' | 'unavailable'>('checking');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [loadingWorkspace, setLoadingWorkspace] = useState(false);
  const [workspaceError, setWorkspaceError] = useState('');
  const [messageInput, setMessageInput] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [sending, setSending] = useState(false);
  const [streamError, setStreamError] = useState('');
  const [mobileMenu, setMobileMenu] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const conversationRef = useRef<Conversation | null>(null);
  conversationRef.current = conversation;
  const orderedModels = useMemo(() => models.length ? models : ['qwen3.5:4b'], [models]);

  async function loadWorkspace() {
    setAuthState('checking'); setLoadingWorkspace(true); setWorkspaceError('');
    try {
      const [me, currentHealth] = await Promise.all([getMe(), getHealth()]);
      if (!me.user.authenticated) { setAuthState('expired'); return; }
      setHealth(currentHealth);
      const [modelResult, conversationResult] = await Promise.all([getModels(), getConversations()]);
      setModels(modelResult.models || []); setSelectedModel(modelResult.models?.[0] || 'qwen3.5:4b'); setConversations(conversationResult.conversations || []); setAuthState('ready');
    } catch (cause) {
      if (cause instanceof LuraApiError && cause.code === 'unauthorized') setAuthState('expired');
      else { setAuthState('unavailable'); setWorkspaceError(cause instanceof LuraApiError ? cause.message : 'The local API is unavailable.'); }
    } finally { setLoadingWorkspace(false); }
  }
  useEffect(() => { void loadWorkspace(); }, []);
  useEffect(() => { if (authState === 'ready' && !activeId && conversations.length) void selectConversation(conversations[0].id); }, [authState, activeId, conversations]);

  async function selectConversation(id: string) {
    setActiveId(id); setLoadingConversation(true); setStreamError(''); setMobileMenu(false);
    try { const result = await getConversation(id); setConversation(withMessages(result.conversation)); } catch (cause) { if (cause instanceof LuraApiError && cause.code === 'unauthorized') setAuthState('expired'); else setStreamError(cause instanceof LuraApiError ? cause.message : 'Could not open this conversation.'); } finally { setLoadingConversation(false); }
  }
  async function createNewConversation() {
    setStreamError('');
    try { const result = await createConversation(); const created = withMessages(result.conversation); setConversations((current) => [created, ...current.filter((item) => item.id !== created.id)]); setActiveId(created.id); setConversation(created); setMobileMenu(false); } catch (cause) { if (cause instanceof LuraApiError && cause.code === 'unauthorized') setAuthState('expired'); else setStreamError(cause instanceof LuraApiError ? cause.message : 'Could not create a conversation.'); }
  }
  async function sendMessage(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault(); const content = messageInput.trim(); if (!content || sending) return; setStreamError('');
    let currentConversation = conversationRef.current;
    try {
      if (!currentConversation) { const createdResult = await createConversation(); const created = withMessages(createdResult.conversation); currentConversation = created; setConversations((current) => [created, ...current]); setActiveId(created.id); setConversation(created); }
      if (!currentConversation) return;
      const currentId = currentConversation.id; const userMessage: Message = { role: 'user', content }; const assistantMessage: Message = { role: 'assistant', content: '' };
      setConversation((current) => current ? { ...current, messages: [...current.messages, userMessage, assistantMessage], updated_at: new Date().toISOString() } : current); setMessageInput(''); setSending(true);
      const controller = new AbortController(); abortRef.current = controller;
      await streamMessage(currentId, { content, model: selectedModel || undefined }, (event) => {
        if (event.type === 'token') { const token = event.data.content || ''; setConversation((current) => { if (!current) return current; const messages = [...current.messages]; const last = messages[messages.length - 1]; if (last?.role === 'assistant') messages[messages.length - 1] = { ...last, content: last.content + token }; return { ...current, messages }; }); }
       if (event.type === 'done') { const finalMessage = event.data.message; if (typeof finalMessage === 'string') setConversation((current) => { if (!current) return current; const messages = [...current.messages]; if (messages[messages.length - 1]?.role === 'assistant') messages[messages.length - 1] = { role: 'assistant', content: finalMessage }; return { ...current, messages, updated_at: new Date().toISOString() }; }); }
        if (event.type === 'error') setStreamError(event.data.message || 'The assistant could not finish this response.');
      }, controller.signal);
      const refreshed = await getConversation(currentId); setConversation(withMessages(refreshed.conversation)); const summaries = await getConversations(); setConversations(summaries.conversations || []);
    } catch (cause) { if (cause instanceof DOMException && cause.name === 'AbortError') setStreamError('Response stopped.'); else if (cause instanceof LuraApiError && cause.code === 'unauthorized') setAuthState('expired'); else setStreamError(cause instanceof LuraApiError ? cause.message : 'The response could not be completed.'); } finally { abortRef.current = null; setSending(false); }
  }
  async function handleLogout() { try { await logout(); } finally { setLocation('/login'); } }

  if (authState === 'checking') return <div className="flex min-h-[100dvh] items-center justify-center bg-[hsl(var(--background))] p-6" data-testid="loading-workspace"><div className="w-full max-w-sm space-y-3"><div className="h-3 w-24 animate-pulse rounded bg-[hsl(var(--muted))]" /><div className="h-10 w-64 animate-pulse rounded bg-[hsl(var(--muted))]" /><div className="h-24 rounded-xl bg-[hsl(var(--muted)/.7)] animate-pulse" /></div></div>;
  if (authState === 'expired') return <main className="flex min-h-[100dvh] items-center justify-center bg-[hsl(var(--background))] p-6"><div className="w-full max-w-md rounded-2xl border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-7 shadow-[var(--shadow-md)]" data-testid="status-session-expired"><BrandMark compact={false} /><p className="mt-10 font-mono text-[10px] uppercase tracking-[.16em] text-[hsl(var(--accent))]">Session closed</p><h1 className="mt-3 text-2xl font-semibold tracking-[-.04em]">Sign in to reopen Lura.</h1><p className="mt-3 text-sm leading-6 text-[hsl(var(--muted-foreground))]">Your local session is no longer valid. Your conversations remain on the Lura service.</p><button type="button" onClick={() => setLocation('/login')} className="mt-7 h-11 rounded-xl bg-[hsl(var(--primary))] px-5 text-sm font-semibold text-[hsl(var(--primary-foreground))]" data-testid="button-session-login">Return to sign in</button></div></main>;
  if (authState === 'unavailable') return <main className="flex min-h-[100dvh] items-center justify-center bg-[hsl(var(--background))] p-6"><div className="w-full max-w-md rounded-2xl border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-7 shadow-[var(--shadow-md)]" data-testid="status-api-unavailable"><div className="mb-5 flex size-11 items-center justify-center rounded-xl bg-[hsl(var(--destructive)/.08)] text-[hsl(var(--destructive))]"><Server size={20} /></div><p className="font-mono text-[10px] uppercase tracking-[.16em] text-[hsl(var(--destructive))]">No local signal</p><h1 className="mt-3 text-2xl font-semibold tracking-[-.04em]">Lura is out of reach.</h1><p className="mt-3 text-sm leading-6 text-[hsl(var(--muted-foreground))]">{workspaceError || 'The local API did not respond.'}</p><button type="button" onClick={() => void loadWorkspace()} className="mt-7 inline-flex h-11 items-center gap-2 rounded-xl bg-[hsl(var(--primary))] px-5 text-sm font-semibold text-[hsl(var(--primary-foreground))]" data-testid="button-retry-workspace"><RefreshCw size={15} /> Reconnect</button></div></main>;
  const hasMessages = Boolean(conversation?.messages?.length);
  return <div className="grain flex min-h-[100dvh] bg-[hsl(var(--background))]" data-testid="chat-workspace"><div className={`fixed inset-0 z-30 bg-[hsl(var(--sidebar)/.5)] transition-opacity md:hidden ${mobileMenu ? 'opacity-100' : 'pointer-events-none opacity-0'}`} onClick={() => setMobileMenu(false)} aria-hidden="true" /><div className={`fixed inset-y-0 left-0 z-40 w-[86vw] max-w-[320px] transition-transform duration-300 md:static md:block md:w-[304px] md:translate-x-0 ${mobileMenu ? 'translate-x-0' : '-translate-x-full'}`}><Sidebar conversations={conversations} activeId={activeId} onSelect={selectConversation} onNew={createNewConversation} onLogout={handleLogout} onClose={() => setMobileMenu(false)} busy={sending} /></div><section className="flex min-w-0 flex-1 flex-col"><header className="flex min-h-[76px] items-center justify-between border-b border-[hsl(var(--border))] bg-[hsl(var(--card)/.74)] px-4 backdrop-blur sm:px-7" data-testid="chat-header"><div className="flex min-w-0 items-center gap-3"><button type="button" onClick={() => setMobileMenu(true)} className="rounded-lg p-2 text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--muted))] md:hidden" aria-label="Open conversations" data-testid="button-open-menu"><Menu size={19} /></button><div className="min-w-0"><div className="flex items-center gap-2"><span className="size-1.5 shrink-0 rounded-full bg-[hsl(var(--primary))]" /><h1 className="truncate text-sm font-semibold sm:text-base" data-testid="text-active-conversation">{conversation?.title || 'New conversation'}</h1></div><p className="mt-1 hidden font-mono text-[9px] uppercase tracking-[.14em] text-[hsl(var(--muted-foreground))] sm:block">Private workspace <span className="mx-1 text-[hsl(var(--border))]">/</span> responses stay local</p></div></div><div className="flex items-center gap-2 sm:gap-4"><ConnectionPill health={health} /><label className="hidden items-center gap-2 rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--background)/.7)] px-2.5 py-2 sm:flex"><span className="font-mono text-[9px] uppercase tracking-[.1em] text-[hsl(var(--muted-foreground))]">model</span><select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} className="max-w-[140px] bg-transparent text-xs font-semibold outline-none" data-testid="select-model">{orderedModels.map((model) => <option key={model} value={model}>{model}</option>)}</select><ChevronDown size={13} className="text-[hsl(var(--muted-foreground))]" /></label></div></header><div className="flex items-center justify-between border-b border-[hsl(var(--border)/.7)] bg-[hsl(var(--background)/.6)] px-5 py-2.5 md:hidden"><span className="font-mono text-[9px] uppercase tracking-[.12em] text-[hsl(var(--muted-foreground))]">active model</span><select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} className="bg-transparent text-xs font-semibold outline-none" data-testid="select-model-mobile">{orderedModels.map((model) => <option key={model} value={model}>{model}</option>)}</select></div>{workspaceError && <div className="mx-4 mt-4 sm:mx-7"><ErrorNotice message={workspaceError} onRetry={() => void loadWorkspace()} /></div>}{streamError && <div className="mx-4 mt-4 sm:mx-7"><ErrorNotice message={streamError} /></div>}<div className="min-h-0 flex-1 overflow-y-auto">{loadingConversation ? <div className="mx-auto flex max-w-3xl flex-col gap-6 px-5 py-10 sm:px-8" data-testid="loading-conversation"><div className="h-16 w-2/3 animate-pulse rounded-lg bg-[hsl(var(--muted))]" /><div className="ml-auto h-12 w-1/2 animate-pulse rounded-xl bg-[hsl(var(--muted))]" /><div className="h-24 w-3/4 animate-pulse rounded-lg bg-[hsl(var(--muted))]" /></div> : hasMessages ? <div className="mx-auto flex max-w-3xl flex-col gap-7 px-5 py-8 sm:px-8 sm:py-12">{conversation?.messages.map((message, index) => <MessageBubble message={message} index={index} key={`${index}-${message.role}`} />)}{sending && <div className="flex items-center gap-3 pl-10 text-xs text-[hsl(var(--muted-foreground))]" data-testid="status-streaming"><span className="flex gap-1"><span className="size-1.5 rounded-full bg-[hsl(var(--primary))] lura-pulse" /><span className="size-1.5 rounded-full bg-[hsl(var(--primary))] lura-pulse [animation-delay:200ms]" /><span className="size-1.5 rounded-full bg-[hsl(var(--primary))] lura-pulse [animation-delay:400ms]" /></span> Lura is thinking locally</div>}</div> : <EmptyWorkspace onNew={createNewConversation} />}</div><div className="mx-auto w-full max-w-3xl px-4 pb-4 pt-2 sm:px-8 sm:pb-7"><form onSubmit={sendMessage} className="relative rounded-2xl border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-2 shadow-[var(--shadow-sm)] focus-within:border-[hsl(var(--primary)/.5)] focus-within:shadow-[0_6px_20px_hsl(var(--primary)/.08)]" data-testid="form-message"><textarea value={messageInput} onChange={(event) => setMessageInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void sendMessage(); } }} disabled={sending} rows={2} placeholder="Message your local assistant…" className="min-h-[58px] w-full resize-none bg-transparent px-3 py-2 pr-14 text-sm leading-6 outline-none placeholder:text-[hsl(var(--muted-foreground)/.65)] disabled:opacity-60" data-testid="input-message" /><div className="flex items-center justify-between px-2 pb-1 pt-1"><span className="font-mono text-[9px] uppercase tracking-[.1em] text-[hsl(var(--muted-foreground)/.65)]">Enter to send <span className="hidden sm:inline">· Shift + Enter for a new line</span></span>{sending ? <button type="button" onClick={() => abortRef.current?.abort()} className="flex size-9 items-center justify-center rounded-xl border border-[hsl(var(--destructive)/.25)] text-[hsl(var(--destructive))] hover:bg-[hsl(var(--destructive)/.08)]" aria-label="Stop response" data-testid="button-stop-response"><Square size={14} fill="currentColor" /></button> : <button type="submit" disabled={!messageInput.trim()} className="flex size-9 items-center justify-center rounded-xl bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] transition-transform hover:-translate-y-0.5 disabled:bg-[hsl(var(--muted))] disabled:text-[hsl(var(--muted-foreground))]" aria-label="Send message" data-testid="button-send-message"><ArrowUp size={17} /></button>}</div></form><p className="mt-3 text-center font-mono text-[9px] uppercase tracking-[.1em] text-[hsl(var(--muted-foreground)/.55)]">Lura can make mistakes · verify anything important</p></div></section></div>;
}

function NotFound() {
  const [, setLocation] = useLocation();
  return <main className="flex min-h-[100dvh] items-center justify-center bg-[hsl(var(--background))] p-6"><div className="text-center"><p className="font-mono text-xs uppercase tracking-[.2em] text-[hsl(var(--primary))]">404</p><h1 className="mt-3 text-3xl font-semibold">This channel does not exist.</h1><button type="button" onClick={() => setLocation('/')} className="mt-6 rounded-xl bg-[hsl(var(--primary))] px-5 py-3 text-sm font-semibold text-[hsl(var(--primary-foreground))]" data-testid="button-not-found-home">Open workspace</button></div></main>;
}

function Router() {
  return <Switch><Route path="/login" component={LoginPage} /><Route path="/" component={ChatWorkspace} /><Route component={NotFound} /></Switch>;
}

function App() {
  return <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><Router /></WouterRouter>;
}

export default App;