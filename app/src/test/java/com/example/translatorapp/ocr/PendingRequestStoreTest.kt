package com.example.translatorapp.ocr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class PendingRequestStoreTest {

    private data class Payload(var value: Int)

    @Test
    fun putAndSnapshot_duplicateValuesAndPreventExternalMutation() {
        val store = PendingRequestStore<Payload>(
            maxEntries = 5,
            duplicate = { payload: Payload -> payload.copy() },
            release = {}
        )

        store.put(1L, Payload(7))

        val firstSnapshot = store.snapshot(1L)
        assertNotNull(firstSnapshot)
        firstSnapshot!!.value = 99

        val secondSnapshot = store.snapshot(1L)
        assertEquals(7, secondSnapshot?.value)
    }

    @Test
    fun put_sameRequestId_releasesPreviousEntry() {
        val released = mutableListOf<Payload>()
        val store = PendingRequestStore<Payload>(
            maxEntries = 5,
            duplicate = { payload: Payload -> payload.copy() },
            release = { payload: Payload -> released.add(payload) }
        )

        store.put(9L, Payload(1))
        store.put(9L, Payload(2))

        assertEquals(1, released.size)
        assertEquals(1, released.first().value)
        assertEquals(2, store.snapshot(9L)?.value)
    }

    @Test
    fun put_overCapacity_evictsOldestEntry() {
        val released = mutableListOf<Payload>()
        val store = PendingRequestStore<Payload>(
            maxEntries = 2,
            duplicate = { payload: Payload -> payload.copy() },
            release = { payload: Payload -> released.add(payload) }
        )

        store.put(1L, Payload(10))
        store.put(2L, Payload(20))
        store.put(3L, Payload(30))

        assertNull(store.snapshot(1L))
        assertEquals(20, store.snapshot(2L)?.value)
        assertEquals(30, store.snapshot(3L)?.value)
        assertEquals(listOf(10), released.map { it.value })
    }

    @Test
    fun discardAndClear_releaseStoredEntries() {
        val released = mutableListOf<Payload>()
        val store = PendingRequestStore<Payload>(
            maxEntries = 10,
            duplicate = { payload: Payload -> payload.copy() },
            release = { payload: Payload -> released.add(payload) }
        )

        store.put(1L, Payload(11))
        store.put(2L, Payload(22))

        store.discard(1L)
        assertEquals(listOf(11), released.map { it.value })

        store.clear()
        assertEquals(listOf(11, 22), released.map { it.value })
        assertNull(store.snapshot(2L))
    }
}
