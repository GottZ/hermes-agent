import { JsonRpcGatewayError } from '@hermes/shared'
import { describe, expect, it, vi } from 'vitest'

import { appGuidanceFromError, officialAppDownloadUrl, presenceForCatalogPlugin } from './plugin-app-guidance'

const missing = {
  reason: 'missing_app',
  app_title: 'Godot 4',
  download_url: 'https://godotengine.org/download/macos/',
  target_host: 'backend'
} as const

describe('catalog app preflight', () => {
  it('asks the backend for only the selected catalog name and target profile', async () => {
    const request = vi.fn(async () => ({ presence: [{ name: 'game-editor', ...missing }] }))
    expect(await presenceForCatalogPlugin(request as never, 'game-editor', 'research')).toMatchObject(missing)
    expect(request).toHaveBeenCalledWith('plugins.manage', {
      action: 'presence',
      name: 'game-editor',
      names: ['game-editor'],
      profile: 'research'
    })
  })

  it('treats older backend method errors as unsupported and missing rows as uncertain', async () => {
    expect(
      await presenceForCatalogPlugin(
        vi.fn(async () => {
          throw new JsonRpcGatewayError('unknown action', { code: 4019 })
        }) as never,
        'game-editor',
        'default'
      )
    ).toBeNull()
    expect(
      await presenceForCatalogPlugin(vi.fn(async () => ({ presence: [] })) as never, 'game-editor', 'default')
    ).toEqual({ reason: 'uncertain', app_title: 'game-editor', target_host: 'backend' })
    expect(
      await presenceForCatalogPlugin(
        vi.fn(async () => {
          throw new Error('network unavailable')
        }) as never,
        'game-editor',
        'default'
      )
    ).toEqual({ reason: 'uncertain', app_title: 'game-editor', target_host: 'backend' })
  })

  it('accepts typed install refusal only from the gateway code and backend host', () => {
    expect(appGuidanceFromError(new JsonRpcGatewayError('refused', { code: 5026, data: missing }))).toMatchObject(
      missing
    )
    expect(appGuidanceFromError(new JsonRpcGatewayError('refused', { code: 5000, data: missing }))).toBeNull()
    expect(
      appGuidanceFromError(
        new JsonRpcGatewayError('refused', { code: 5026, data: { ...missing, target_host: 'desktop' } })
      )
    ).toBeNull()
  })

  it('only offers exact allowlisted HTTPS vendor links for actionable blockers', () => {
    expect(officialAppDownloadUrl(missing)).toBe('https://godotengine.org/download/macos/')
    expect(
      officialAppDownloadUrl({ ...missing, download_url: 'https://godotengine.org.evil.test/download/macos/' })
    ).toBeNull()
    expect(officialAppDownloadUrl({ ...missing, download_url: 'http://godotengine.org/download/macos/' })).toBeNull()
    expect(officialAppDownloadUrl({ ...missing, reason: 'unsupported_os' })).toBeNull()
  })
})
