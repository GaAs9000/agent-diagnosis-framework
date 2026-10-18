// Keep native schemas and preparation hooks intact; intercept only execution.
import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline';

globalThis.fetch = () => { throw new Error('Network disabled in validation bridge'); };
const base = dirname(fileURLToPath(import.meta.url));
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const lockBytes = readFileSync(join(base, 'package-lock.json'));
const lock = JSON.parse(lockBytes);
const source = JSON.parse(readFileSync(join(base, 'provenance.json')));
const approvedBytes = readFileSync(join(base, 'approved-modules.json'));
const approved = JSON.parse(approvedBytes);
if (approved.format !== 'c3-pi-approved-modules-v1' || approved.lock_sha256 !== sha(lockBytes)) {
  throw new Error('Approved native module manifest does not match dependency lock');
}
const actualModules = {};
function inspectApprovedModules(directory, relative) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const key = `${relative}/${entry.name}`;
    const path = join(directory, entry.name);
    if (entry.isDirectory()) inspectApprovedModules(path, key);
    else if (entry.isFile()) {
      if (approved.extensions.includes(extname(entry.name))) actualModules[key] = sha(readFileSync(path));
    } else throw new Error(`Unexpected critical dependency filesystem entry: ${key}`);
  }
}
for (const name of approved.packages) inspectApprovedModules(join(base, 'node_modules', name), name);
const expectedKeys = Object.keys(approved.files).sort();
const actualKeys = Object.keys(actualModules).sort();
if (JSON.stringify(expectedKeys) !== JSON.stringify(actualKeys) ||
    expectedKeys.some((key) => actualModules[key] !== approved.files[key])) {
  throw new Error('Installed critical native modules differ from approved manifest');
}
// Bind the identity to installed bytes as well as registry archive integrities.
const installedHash = createHash('sha256');
let installedFiles = 0;
function hashInstalledTree(directory, relative = '') {
  for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name, 'en'))) {
    if (entry.name === '.bin' || entry.name === '.package-lock.json') continue;
    const path = join(directory, entry.name);
    const key = `${relative}/${entry.name}`;
    if (entry.isDirectory()) hashInstalledTree(path, key);
    else if (entry.isFile()) {
      installedHash.update(`${key}\0${sha(readFileSync(path))}\n`);
      installedFiles++;
    } else throw new Error(`Unexpected dependency filesystem entry: ${key}`);
  }
}
hashInstalledTree(join(base, 'node_modules'));
const packages = {};
for (const name of ['pi-agent-core', 'pi-ai', 'pi-coding-agent', 'pi-tui']) {
  const id = `@mariozechner/${name}`;
  const entry = fileURLToPath(import.meta.resolve(id));
  const root = dirname(dirname(entry));
  const metadata = JSON.parse(readFileSync(join(root, 'package.json')));
  const pinned = lock.packages[`node_modules/${id}`];
  if (metadata.version !== source.version || pinned?.version !== source.version || !pinned.integrity) {
    throw new Error(`Unpinned Pi dependency: ${id}`);
  }
  packages[id] = { version: metadata.version, integrity: pinned.integrity };
}
const provenance = {
  ...source, packages, lock_sha256: sha(lockBytes),
  bridge_sha256: sha(readFileSync(fileURLToPath(import.meta.url))),
  approved_modules_sha256: sha(approvedBytes), approved_modules_verified: true,
  installed_tree_sha256: installedHash.digest('hex'), installed_files: installedFiles,
  node_version: process.version,
};
provenance.dependency_identity = sha(JSON.stringify(provenance));
const { runAgentLoop } = await import('@mariozechner/pi-agent-core');
// Import tool modules directly to avoid coding-agent CLI/session initialization.
const codingRoot = dirname(fileURLToPath(import.meta.resolve('@mariozechner/pi-coding-agent')));
const tools = [];
let executions = 0;
for (const name of ['read', 'bash', 'edit', 'write']) {
  const module = await import(pathToFileURL(join(codingRoot, 'core', 'tools', `${name}.js`)));
  const factory = `create${name[0].toUpperCase()}${name.slice(1)}Tool`;
  const tool = module[factory]('/workspace');
  tool.execute = async () => { executions++; throw new Error('Forbidden native execution'); };
  tools.push(tool);
}
const clone = (value) => structuredClone(value);

async function validateBatch(calls) {
  const raw = calls.map((call) => clone(call.arguments));
  const validated = new Map();
  let responses = 0;
  const assistant = {
    role: 'assistant', api: 'validation-only', provider: 'validation-only', model: 'recorded',
    content: calls.map((call) => ({ type: 'toolCall', ...clone(call) })),
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    stopReason: 'toolUse', timestamp: 0,
  };
  const messages = await runAgentLoop([], { systemPrompt: '', messages: [], tools }, {
    model: { id: 'recorded', provider: 'validation-only', api: 'validation-only' },
    convertToLlm: (messages) => messages,
    toolExecution: 'sequential',
    beforeToolCall: ({ toolCall, args }) => {
      validated.set(toolCall.id, clone(args));
      return { block: true, reason: 'C3_VALIDATION_ONLY_EXECUTION_BLOCKED' };
    },
    shouldStopAfterTurn: () => true,
  }, () => {}, undefined, () => {
    responses++;
    if (responses !== 1) throw new Error('Unexpected additional assistant response');
    return {
      async *[Symbol.asyncIterator]() { yield { type: 'done', message: assistant }; },
      result: async () => assistant,
    };
  });
  if (executions || responses !== 1) throw new Error('Validation-only invariant violated');
  const results = messages.filter((message) => message.role === 'toolResult');
  if (results.length !== calls.length) throw new Error('Native result count mismatch');
  return results.map((result, index) => {
    const ok = validated.has(calls[index].id);
    return {
      ok, raw_arguments: raw[index],
      validated_arguments: ok ? validated.get(calls[index].id) : null,
      // Successful validation has no execution result. Never expose the blocking sentinel.
      native_result: ok ? null : { content: result.content, details: result.details, isError: result.isError },
      provenance,
      telemetry: { actor_requests: 0, native_loop_observed_responses: responses, native_tool_executions: executions },
    };
  });
}

const send = (value) => {
  const line = JSON.stringify(value);
  if (Buffer.byteLength(line) > 1024 * 1024) throw new Error('Bridge response exceeds limit');
  process.stdout.write(`${line}\n`);
};
send({ ready: true, provenance });
for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
  try {
    if (Buffer.byteLength(line) > 1024 * 1024) throw new Error('Bridge request exceeds limit');
    const request = JSON.parse(line);
    if (request.op === 'schemas') {
      send({ schemas: tools.map(({ name, description, parameters }) => ({ name, description, parameters })), provenance });
    } else if (request.op === 'validate') {
      send((await validateBatch([{ name: request.name, arguments: request.arguments, id: request.call_id }]))[0]);
    } else if (request.op === 'validate_batch') {
      if (!Array.isArray(request.calls) || request.calls.length < 1 || request.calls.length > 128 ||
          new Set(request.calls.map((call) => call.id)).size !== request.calls.length) {
        throw new Error('Invalid batch envelope');
      }
      send({ results: await validateBatch(request.calls), provenance });
    } else throw new Error('Unknown bridge operation');
  } catch (error) {
    send({ infrastructure_error: error instanceof Error ? error.message : String(error) });
    process.exitCode = 1;
    break;
  }
}
