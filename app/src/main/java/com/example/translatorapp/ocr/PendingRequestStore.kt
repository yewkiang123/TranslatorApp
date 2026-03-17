package com.example.translatorapp.ocr

internal class PendingRequestStore<T>(
    private val maxEntries: Int,
    private val duplicate: (T) -> T,
    private val release: (T) -> Unit
) {
    private val entries = LinkedHashMap<Long, T>()

    fun put(requestId: Long, value: T) {
        entries.remove(requestId)?.let(release)
        entries[requestId] = duplicate(value)
        trimToMaxEntries()
    }

    fun snapshot(requestId: Long): T? {
        val value = entries[requestId] ?: return null
        return duplicate(value)
    }

    fun discard(requestId: Long) {
        entries.remove(requestId)?.let(release)
    }

    fun clear() {
        entries.values.forEach(release)
        entries.clear()
    }

    private fun trimToMaxEntries() {
        while (entries.size > maxEntries) {
            val oldestRequestId = entries.entries.firstOrNull()?.key ?: break
            entries.remove(oldestRequestId)?.let(release)
        }
    }
}
