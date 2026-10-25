import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const { routedRequest, lookup } = vi.hoisted(() => ({
  routedRequest: vi.fn(async (_connection: null | string, _profile: string, _method: string,
    params: Record<string, unknown>): Promise<unknown> =>
    params.action === 'presence'
      ? { presence: [{ name: 'nvidia-app', title: 'NVIDIA App', state: 'missing_app', sentence: 'needs NVIDIA App' }] }
      : { plugins: [] }),
  lookup: vi.fn(async () => ({ ok: true, entry: {
    name: 'nvidia-app', repo: 'https://github.com/NousResearch/hermes-nvidia', sha: 'a'.repeat(40)
  } }))
}))

vi.mock('@/app/chat/session-view', async () => {
  const { atom } = await import('nanostores')
  const view = { $storedId: atom('chat-1') }
  return { useSessionView: () => view }
})
vi.mock('@/components/assistant-ui/use-connection-owner', () => {
  const owner = { connectionId: 'local', profile: 'workbot' }
  return { useConnectionOwner: () => owner }
})
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: (...args: Parameters<typeof routedRequest>) => routedRequest(...args)
}))
vi.mock('@/lib/plugin-catalog', async original => ({
  ...(await original<Record<string, unknown>>()),
  lookupPluginCatalogEntry: (...args: Parameters<typeof lookup>) => lookup(...args)
}))

import { queryClient } from '@/lib/query-client'
import {
  $agentPlugins, $agentPluginsByOwner, $agentPluginsConnectionId, $agentPluginsErrorByOwner,
  $agentPluginsLoadingByOwner,
  $agentPluginsProfile
} from '@/store/agent-plugins'
import { $pluginInstallRequest, closePluginInstallRequest } from '@/store/plugin-install-request'

import { PluginsCard } from './plugins-card'

beforeEach(() => {
  queryClient.clear()
  $agentPlugins.set([])
  $agentPluginsByOwner.set({})
  $agentPluginsErrorByOwner.set({})
  $agentPluginsLoadingByOwner.set({})
  $agentPluginsProfile.set(undefined)
  $agentPluginsConnectionId.set(undefined)
  closePluginInstallRequest()
  routedRequest.mockReset()
  routedRequest.mockImplementation(async (_connection, _profile, _method, params) =>
    params.action === 'presence'
      ? { presence: [{ name: 'nvidia-app', title: 'NVIDIA App', state: 'missing_app', sentence: 'needs NVIDIA App' }] }
      : { plugins: [] })
  lookup.mockClear()
})

afterEach(() => {
  cleanup()
  queryClient.clear()
})

it('loads the chat owner inventory and pins the full owner when installing from setup', async () => {
  render(
    <QueryClientProvider client={queryClient}>
      <PluginsCard names={['nvidia-app']} onContinue={vi.fn()} />
    </QueryClientProvider>
  )
  fireEvent.click(await screen.findByRole('button', { name: /NVIDIA App/ }))
  await waitFor(() => expect($pluginInstallRequest.get()?.profile).toEqual({ connectionId: 'local', profile: 'workbot' }))
  expect(routedRequest).toHaveBeenCalledWith('local', 'workbot', 'plugins.manage',
    expect.objectContaining({ action: 'list', profile: 'workbot' }), undefined)
  expect(routedRequest).toHaveBeenCalledWith('local', 'workbot', 'plugins.manage',
    expect.objectContaining({ action: 'presence', names: ['nvidia-app'] }))
  expect(lookup).toHaveBeenCalledWith('nvidia-app')
})

it('requires an owner inventory retry rather than treating a failed list as not installed', async () => {
  routedRequest.mockImplementation(async (_connection, _profile, _method, params) =>
    params.action === 'presence'
      ? { presence: [{ name: 'nvidia-app', title: 'NVIDIA App', state: 'missing_app' }] }
      : Promise.reject(new Error('inventory unavailable')))
  render(
    <QueryClientProvider client={queryClient}>
      <PluginsCard names={['nvidia-app']} onContinue={vi.fn()} />
    </QueryClientProvider>
  )
  expect((await screen.findByRole('alert')).textContent).toContain('inventory unavailable')
  expect(screen.queryByRole('button', { name: /NVIDIA App/ })).toBeNull()
  expect(lookup).not.toHaveBeenCalled()
  routedRequest.mockImplementation(async (_connection, _profile, _method, params) =>
    params.action === 'presence'
      ? { presence: [{ name: 'nvidia-app', title: 'NVIDIA App', state: 'missing_app' }] }
      : { plugins: [{ name: 'nvidia-app', catalog_name: 'nvidia-app' }] })
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
  expect(await screen.findByText('Installed')).toBeTruthy()
  expect(lookup).not.toHaveBeenCalled()
})

it('shows a retry instead of no plugins when the backend presence query fails', async () => {
  routedRequest.mockImplementation(async (_connection, _profile, _method, params) =>
    params.action === 'presence'
      ? Promise.reject(new Error('presence unavailable'))
      : { plugins: [] })
  render(
    <QueryClientProvider client={queryClient}>
      <PluginsCard names={['nvidia-app']} onContinue={vi.fn()} />
    </QueryClientProvider>
  )
  expect((await screen.findByRole('alert')).textContent).toContain('presence unavailable')
  routedRequest.mockImplementation(async (_connection, _profile, _method, params) =>
    params.action === 'presence'
      ? { presence: [{ name: 'nvidia-app', title: 'NVIDIA App', state: 'missing_app' }] }
      : { plugins: [] })
  fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
  expect(await screen.findByRole('button', { name: /NVIDIA App/ })).toBeTruthy()
})
