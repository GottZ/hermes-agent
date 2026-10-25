import type { ProfileScope } from '@/hermes'

import type { GatewayRequest } from './agent-plugins'
import { requestGatewayForAgent } from './gateway'

/** Pin all plugin RPCs to the selected backend and profile, not the foreground chat. */
export function pluginOwnerRequest(scope: ProfileScope, ambient: GatewayRequest): GatewayRequest {
  if (!scope || typeof scope !== 'object' || !scope.connectionId?.trim()) {
    return ambient
  }

  const connectionId = scope.connectionId.trim()
  return <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) =>
    requestGatewayForAgent<T>(
      connectionId,
      String(params?.profile ?? scope.profile ?? 'default'),
      method,
      params,
      timeoutMs
    )
}
