import type { ProfileScope } from '@/hermes'
import { translateNow } from '@/i18n'
import { lookupPluginCatalogEntry, type PluginCatalogEntry, type PluginCatalogLookupError } from '@/lib/plugin-catalog'

import { notify } from './notifications'
import { openPluginInstallRequest, type PluginInstallRequest } from './plugin-install-request'

/**
 * THE way a curated-catalog pick reaches the Install Plugin dialog. The
 * Plugins tab's embedded picker and the `hermes://plugin/install?catalog=`
 * deep link both land here, so a link opens exactly the reviewed/pinned
 * dialog an in-app pick does: `catalogName` makes the backend resolve the
 * pinned SHA and record provenance; `repo#subdir` is what the dialog inspects.
 */
export function openCatalogPluginInstall(
  entry: PluginCatalogEntry,
  profile: ProfileScope,
  options: Pick<PluginInstallRequest, 'legacyHint' | 'stayAfterInstall' | 'tier' | 'title'> = {}
): void {
  // The shared inventory has only a profile label, not a connection owner.
  // Never treat another backend's matching row as proof this owner has the plugin.
  openPluginInstallRequest({
    catalogName: entry.name,
    profile,
    repo: entry.subdir ? `${entry.repo}#${entry.subdir}` : entry.repo,
    sha: entry.sha,
    summary: entry.summary,
    tier: entry.tier,
    ...options
  })
}

const DEEP_LINK_ERROR_KEYS: Record<PluginCatalogLookupError, string> = {
  invalid_name: 'skills.plugins.deepLinkCatalogInvalidName',
  unavailable: 'skills.plugins.deepLinkCatalogUnavailable',
  unknown: 'skills.plugins.deepLinkCatalogUnknown'
}

/** Same actionable lookup failure on deep links and owner-scoped setup cards. */
export function notifyPluginCatalogLookupError(name: string, error: PluginCatalogLookupError): void {
  notify({
    kind: 'error',
    title: translateNow('skills.plugins.deepLinkErrorTitle'),
    message: translateNow(DEEP_LINK_ERROR_KEYS[error], name.trim())
  })
}

/**
 * `hermes://plugin/install?catalog=<name>`: resolve the name against the live
 * catalog and open the dialog in catalog mode for the active profile. Any
 * failure (bad name, catalog unreachable, name not listed) is a clear error
 * toast — the string is never reinterpreted as a git path.
 */
export async function requestPluginCatalogInstallFromDeepLink(
  name: string,
  lookup: typeof lookupPluginCatalogEntry = lookupPluginCatalogEntry
): Promise<void> {
  const result = await lookup(name)

  if (!result.ok) {
    notifyPluginCatalogLookupError(name, result.error)

    return
  }

  openCatalogPluginInstall(result.entry, null)
}
