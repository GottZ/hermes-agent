import { JsonRpcGatewayError } from '@hermes/shared'

import type { GatewayRequest } from './agent-plugins'

export type AppBlockerReason = 'missing_app' | 'version_too_old' | 'unsupported_os' | 'unsupported_gpu'
export type AppPresenceReason = AppBlockerReason | 'available' | 'app_not_running' | 'uncertain'

export interface AppGuidance {
  reason: AppPresenceReason
  app_title: string
  required_version?: string | null
  found_version?: string | null
  download_url?: string | null
  target_host: 'backend'
}

const REASONS = new Set<AppPresenceReason>([
  'missing_app',
  'version_too_old',
  'unsupported_os',
  'unsupported_gpu',
  'available',
  'app_not_running',
  'uncertain'
])

function parseAppGuidance(value: unknown): AppGuidance | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const data = value as Record<string, unknown>

  if (
    data.target_host !== 'backend' ||
    typeof data.app_title !== 'string' ||
    !data.app_title.trim() ||
    typeof data.reason !== 'string' ||
    !REASONS.has(data.reason as AppPresenceReason)
  ) {
    return null
  }

  return {
    reason: data.reason as AppPresenceReason,
    app_title: data.app_title,
    target_host: 'backend',
    required_version: typeof data.required_version === 'string' ? data.required_version : null,
    found_version: typeof data.found_version === 'string' ? data.found_version : null,
    download_url: typeof data.download_url === 'string' ? data.download_url : null
  }
}

/** Only catalog picks use backend app discovery. Null means an older backend has no preflight. */
export async function presenceForCatalogPlugin(
  request: GatewayRequest,
  name: string,
  profile: string
): Promise<AppGuidance | null> {
  try {
    // `names` is the current batch contract; `name` allows single-name implementations.
    const result = await request<{ presence?: unknown }>('plugins.manage', {
      action: 'presence',
      name,
      names: [name],
      profile
    })

    if (!Array.isArray(result?.presence)) {
      return null
    }

    const row = result.presence.find(candidate => candidate?.name === name)

    return parseAppGuidance(row) ?? { reason: 'uncertain', app_title: name, target_host: 'backend' }
  } catch (error) {
    if (
      error instanceof JsonRpcGatewayError &&
      (error.code === -32601 || error.code === 4019 || error.code === -32602)
    ) {
      return null
    }

    // A failed read cannot assert a missing application; offer an honest uncertain state.
    return { reason: 'uncertain', app_title: name, target_host: 'backend' }
  }
}

/** An install race can re-introduce a typed refusal after a successful preflight. */
export function appGuidanceFromError(error: unknown): AppGuidance | null {
  return error instanceof JsonRpcGatewayError && error.code === 5026 ? parseAppGuidance(error.data) : null
}

const VENDOR_URLS = new Map([
  ['NVIDIA App', 'https://www.nvidia.com/en-us/software/nvidia-app/'],
  ['NVIDIA Broadcast', 'https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/'],
  ['Godot 4', 'https://godotengine.org/download/macos/']
])

/** Vendor-controlled, HTTPS, exact URLs only — never trust plugin docs_url or a supplied hostname. */
export function officialAppDownloadUrl(guidance: AppGuidance): string | null {
  if (guidance.target_host !== 'backend' || !['missing_app', 'version_too_old'].includes(guidance.reason)) {
    return null
  }

  const allowed = VENDOR_URLS.get(guidance.app_title)

  return guidance.download_url === allowed && allowed?.startsWith('https://') ? allowed : null
}

export function isAppBlocked(guidance: AppGuidance | null): boolean {
  return (
    guidance !== null &&
    ['missing_app', 'version_too_old', 'unsupported_os', 'unsupported_gpu'].includes(guidance.reason)
  )
}
