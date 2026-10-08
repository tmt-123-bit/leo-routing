/*
 * PurgeSrpfQueue: the packet-management mechanism queue family for Hypatia.
 *
 * Continuous-time form of the slot-based family: at dequeue time, packets
 * whose remaining deadline budget cannot cover their remaining hop distance
 * (optimistic lower bound: hop count x minimal per-hop latency, plus a
 * one-hop safety margin) are dropped instead of transmitted. The dequeue
 * ordering and dropping behavior follow the configured mode:
 *
 *   droptail       plain FIFO (base arm)
 *   edf            earliest absolute deadline first
 *   class_priority traffic-class priority, deadline within class
 *   lcfs           newest packet first
 *   codel          CoDel-style age drop (age at dequeue > target -> drop)
 *   red            random early drop on enqueue by queue depth
 *   purge          feasibility purge, FIFO ordering
 *   purge_srpf     feasibility purge, fewest remaining hops first
 *   purge_edf      feasibility purge + EDF ordering
 *   purge_lcfs     feasibility purge + newest-first ordering
 *
 * Remaining hops come from SatnetHopOracle, which chases the currently
 * installed per-interval forwarding state (satgenpy fstate files).
 *
 * Purge is applied at dequeue (never refuses at enqueue): refusing at
 * enqueue propagates a send error through ns-3's UDP/IP error path and
 * stalls the entire flow (observed empirically), and the device's
 * Dequeue-after-Enqueue contract requires a possible null return (the
 * companion net-device patch adds the null guard).
 */

#ifndef PURGE_SRPF_QUEUE_H
#define PURGE_SRPF_QUEUE_H

#include <deque>
#include "ns3/queue.h"
#include "ns3/queue-size.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"
#include "ns3/type-name.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/random-variable-stream.h"
#include "deadline-tag.h"
#include "satnet-hop-oracle.h"

namespace ns3 {

enum MechQueueMode {
    MODE_DROPTAIL = 0,
    MODE_EDF = 1,
    MODE_CLASS_PRIORITY = 2,
    MODE_LCFS = 3,
    MODE_CODEL = 4,
    MODE_RED = 5,
    MODE_PURGE = 6,
    MODE_PURGE_SRPF = 7,
    MODE_PURGE_EDF = 8,
    MODE_PURGE_LCFS = 9,
    MODE_DROPFRONT = 10
};

template <typename Item>
class PurgeSrpfQueue : public Queue<Item>
{
public:
    static TypeId GetTypeId (void);

    PurgeSrpfQueue ();
    virtual ~PurgeSrpfQueue ();

    virtual bool Enqueue (Ptr<Item> item);
    virtual Ptr<Item> Dequeue (void);
    virtual Ptr<Item> Remove (void);
    virtual Ptr<const Item> Peek (void) const;

    void SetNodeId (int32_t node_id);
    uint64_t GetPurgedCount (void) const;
    static uint64_t GetTotalPurged (void);
    static uint64_t GetTotalManaged (void);
    static uint64_t GetStat (const std::string& name) {
        if (name == "rejected") return s_rejected;
        if (name == "enqueue_false") return s_enqueueFalse;
        if (name == "dequeue_null") return s_dequeueNull;
        if (name == "peek_hits") return s_peekHits;
        if (name == "hops_sum") return s_hopsSum;
        if (name == "hops_neg") return s_hopsNeg;
        return 0;
    }

private:
    using Queue<Item>::begin;
    using Queue<Item>::end;
    using Queue<Item>::DoEnqueue;
    using Queue<Item>::DoDequeue;
    using Queue<Item>::DoRemove;
    using Queue<Item>::DoPeek;

    // Priority key for the dequeue pick (smaller = served earlier).
    // Tagged packets are keyed by mode; untagged traffic always keys 0
    // (plain FIFO ahead of tagged ones, never starved).
    uint64_t KeyOf (Ptr<const Item> item);
    // Feasibility purge criterion (purge* modes only).
    bool IsInfeasible (Ptr<const Item> item);
    // CoDel-style age criterion.
    bool IsAgedOut (Ptr<const Item> item);

    int32_t m_nodeId;
    uint64_t m_purgedCount;
    uint64_t m_managedCount;   // mode-specific interventions (codel/red drops)
    uint32_t m_mode;
    uint64_t m_minHopLatencyNs;
    uint64_t m_codelTargetNs;
    uint32_t m_redMinTh;
    uint32_t m_redMaxTh;
    double m_redMaxP;
    uint32_t m_dropfrontTh;
    Ptr<UniformRandomVariable> m_redRng;

    static uint64_t s_totalPurged;
    static uint64_t s_totalManaged;
    static uint64_t s_rejected;
    static uint64_t s_enqueueFalse;
    static uint64_t s_dequeueNull;
    static uint64_t s_peekHits;
    static uint64_t s_hopsSum;
    static uint64_t s_hopsNeg;
};

template <typename Item>
uint64_t PurgeSrpfQueue<Item>::s_totalPurged = 0;
template <typename Item>
uint64_t PurgeSrpfQueue<Item>::s_totalManaged = 0;
template <typename Item>
uint64_t PurgeSrpfQueue<Item>::s_rejected = 0;
template <typename Item>
uint64_t PurgeSrpfQueue<Item>::s_enqueueFalse = 0;
template <typename Item>
uint64_t PurgeSrpfQueue<Item>::s_dequeueNull = 0;
template <typename Item>
uint64_t PurgeSrpfQueue<Item>::s_peekHits = 0;
template <typename Item>
uint64_t PurgeSrpfQueue<Item>::s_hopsSum = 0;
template <typename Item>
uint64_t PurgeSrpfQueue<Item>::s_hopsNeg = 0;

template <typename Item>
TypeId
PurgeSrpfQueue<Item>::GetTypeId (void)
{
    static TypeId tid = TypeId (("ns3::PurgeSrpfQueue<" + GetTypeParamName<PurgeSrpfQueue<Item> > () + ">").c_str ())
        .SetParent<Queue<Item> > ()
        .SetGroupName ("SatelliteNetwork")
        .template AddConstructor<PurgeSrpfQueue<Item> > ()
        .AddAttribute ("MaxSize",
                       "The max queue size",
                       QueueSizeValue (QueueSize ("100p")),
                       MakeQueueSizeAccessor (&QueueBase::SetMaxSize,
                                              &QueueBase::GetMaxSize),
                       MakeQueueSizeChecker ())
        .AddAttribute ("Mode",
                       "0 droptail, 1 edf, 2 class_priority, 3 lcfs, 4 codel, "
                       "5 red, 6 purge, 7 purge_srpf, 8 purge_edf, 9 purge_lcfs",
                       UintegerValue (7),
                       MakeUintegerAccessor (&PurgeSrpfQueue<Item>::m_mode),
                       MakeUintegerChecker<uint32_t> ())
        .AddAttribute ("MinHopLatencyNs",
                       "Optimistic lower bound on the latency of one hop; "
                       "used by the purge criterion",
                       UintegerValue (3000000),
                       MakeUintegerAccessor (&PurgeSrpfQueue<Item>::m_minHopLatencyNs),
                       MakeUintegerChecker<uint64_t> ())
        .AddAttribute ("CodelTargetNs",
                       "CoDel-style age target for the codel arm",
                       UintegerValue (12000000),
                       MakeUintegerAccessor (&PurgeSrpfQueue<Item>::m_codelTargetNs),
                       MakeUintegerChecker<uint64_t> ())
        .AddAttribute ("RedMinTh",
                       "RED minimum queue threshold (packets)",
                       UintegerValue (12),
                       MakeUintegerAccessor (&PurgeSrpfQueue<Item>::m_redMinTh),
                       MakeUintegerChecker<uint32_t> ())
        .AddAttribute ("RedMaxTh",
                       "RED maximum queue threshold (packets)",
                       UintegerValue (24),
                       MakeUintegerAccessor (&PurgeSrpfQueue<Item>::m_redMaxTh),
                       MakeUintegerChecker<uint32_t> ())
        .AddAttribute ("RedMaxP",
                       "RED maximum drop probability",
                       DoubleValue (0.1),
                       MakeDoubleAccessor (&PurgeSrpfQueue<Item>::m_redMaxP),
                       MakeDoubleChecker<double> (0.0, 1.0))
        .AddAttribute ("DropfrontTh",
                       "Drop-front pressure threshold (packets)",
                       UintegerValue (16),
                       MakeUintegerAccessor (&PurgeSrpfQueue<Item>::m_dropfrontTh),
                       MakeUintegerChecker<uint32_t> ())
    ;
    return tid;
}

template <typename Item>
PurgeSrpfQueue<Item>::PurgeSrpfQueue () :
    Queue<Item> (),
    m_nodeId (-1),
    m_purgedCount (0),
    m_managedCount (0),
    m_mode (7),
    m_minHopLatencyNs (3000000),
    m_codelTargetNs (12000000),
    m_redMinTh (12),
    m_redMaxTh (24),
    m_redMaxP (0.1),
    m_dropfrontTh (16)
{
    m_redRng = CreateObject<UniformRandomVariable> ();
    m_redRng->SetAttribute ("Min", DoubleValue (0.0));
    m_redRng->SetAttribute ("Max", DoubleValue (1.0));
}

template <typename Item>
PurgeSrpfQueue<Item>::~PurgeSrpfQueue ()
{}

template <typename Item>
bool
PurgeSrpfQueue<Item>::Enqueue (Ptr<Item> item)
{
    // RED arm: probabilistic early drop by instantaneous queue depth.
    // Purge-family interventions happen at dequeue (see Dequeue): refusing
    // here would propagate a send error through ns-3's UDP/IP error path,
    // which stalls the entire flow after the first refusal.
    if (m_mode == MODE_RED) {
        uint32_t len = Queue<Item>::GetNPackets ();
        DeadlineTag tag;
        bool tagged = item->PeekPacketTag (tag);
        if (tagged && len >= m_redMinTh) {
            double p = m_redMaxP;
            if (m_redMaxTh > m_redMinTh) {
                p = m_redMaxP * ((double) len - (double) m_redMinTh)
                    / ((double) m_redMaxTh - (double) m_redMinTh);
            }
            if (m_redRng->GetValue () < p) {
                s_rejected++;
                m_managedCount++;
                s_totalManaged++;
                Queue<Item>::DropBeforeEnqueue (item);
                return false;
            }
        }
    }
    bool ok = DoEnqueue (end (), item);
    if (!ok) s_enqueueFalse++;
    return ok;
}

template <typename Item>
uint64_t
PurgeSrpfQueue<Item>::KeyOf (Ptr<const Item> item)
{
    DeadlineTag tag;
    if (!item->PeekPacketTag (tag)) {
        return 0;  // untagged traffic: plain FIFO, highest priority
    }
    switch (m_mode) {
        case MODE_EDF:
        case MODE_PURGE_EDF:
            // earliest absolute deadline first (deadlines are ns since epoch;
            // clamp so the key stays within uint64)
            return tag.GetDeadlineNs () / 1000ULL;
        case MODE_CLASS_PRIORITY:
        case MODE_PURGE:  // purge keeps FIFO; class key only in class arm
            // class first, then deadline
            return ((uint64_t) tag.GetTrafficClass () << 40)
                 | (tag.GetDeadlineNs () / 1000000ULL);
        case MODE_LCFS:
        case MODE_PURGE_LCFS:
            // newest first: invert send time (bounded window)
            return (uint64_t)(0xFFFFFFFFFFFFFFFFULL
                              - (tag.GetSendNs () / 1000ULL));
        case MODE_PURGE_SRPF: {
            int64_t hops = SatnetHopOracle::Get ()->HopDistance (m_nodeId, (int32_t) tag.GetDstNodeId ());
            if (hops < 0) {
                return (uint64_t) 1000000;
            }
            return (uint64_t) hops;
        }
        default:
            return 0;  // FIFO modes
    }
}

template <typename Item>
bool
PurgeSrpfQueue<Item>::IsInfeasible (Ptr<const Item> item)
{
    if (m_nodeId < 0) return false;
    DeadlineTag tag;
    if (!item->PeekPacketTag (tag)) {
        return false;
    }
    s_peekHits++;
    int64_t hops = SatnetHopOracle::Get ()->HopDistance (m_nodeId, (int32_t) tag.GetDstNodeId ());
    if (hops < 0) s_hopsNeg++;
    s_hopsSum += (uint64_t)(hops > 0 ? hops : 0);
    if (hops > 0) {
        uint64_t now = (uint64_t) Simulator::Now ().GetNanoSeconds ();
        uint64_t need = ((uint64_t) hops + 1) * m_minHopLatencyNs;  // +1 hop safety margin
        if (now + need > tag.GetDeadlineNs ()) {
            return true;
        }
    }
    return false;
}

template <typename Item>
bool
PurgeSrpfQueue<Item>::IsAgedOut (Ptr<const Item> item)
{
    DeadlineTag tag;
    if (!item->PeekPacketTag (tag)) {
        return false;
    }
    uint64_t now = (uint64_t) Simulator::Now ().GetNanoSeconds ();
    return now > tag.GetSendNs () + m_codelTargetNs;
}

template <typename Item>
Ptr<Item>
PurgeSrpfQueue<Item>::Dequeue (void)
{
    // Dequeue-time interventions: purge the certainly-infeasible, drop the
    // aged-out (codel), then serve by the mode's ordering key.
    bool purgeOn = (m_mode == MODE_PURGE || m_mode == MODE_PURGE_SRPF
                    || m_mode == MODE_PURGE_EDF || m_mode == MODE_PURGE_LCFS);
    bool codelOn = (m_mode == MODE_CODEL);
    bool dropfrontOn = (m_mode == MODE_DROPFRONT);
    bool orderOn = (m_mode == MODE_EDF || m_mode == MODE_CLASS_PRIORITY
                    || m_mode == MODE_LCFS || m_mode == MODE_PURGE_SRPF
                    || m_mode == MODE_PURGE_EDF || m_mode == MODE_PURGE_LCFS);

    while (!Queue<Item>::IsEmpty ()) {
        // pick the next packet per ordering (front for FIFO modes)
        Ptr<Item> item;
        if (orderOn) {
            auto best = begin ();
            uint64_t bestKey = KeyOf (*best);
            for (auto it = begin (); it != end (); ++it) {
                uint64_t k = KeyOf (*it);
                if (k < bestKey) {
                    bestKey = k;
                    best = it;
                }
            }
            item = DoRemove (best);
        } else {
            item = DoDequeue (begin ());
        }
        if (item == 0) {
            s_dequeueNull++;
            return item;
        }
        if (purgeOn && IsInfeasible (item)) {
            s_rejected++;
            Queue<Item>::DropAfterDequeue (item);
            m_purgedCount++;
            s_totalPurged++;
            continue;  // try the next candidate
        }
        if (codelOn && IsAgedOut (item)) {
            s_rejected++;
            Queue<Item>::DropAfterDequeue (item);
            m_managedCount++;
            s_totalManaged++;
            continue;
        }
        if (dropfrontOn) {
            // Pressure drop: while depth exceeds the threshold, drop the
            // head-of-line packet (the oldest) instead of transmitting it.
            if (Queue<Item>::GetNPackets () + 1 > m_dropfrontTh) {
                s_rejected++;
                Queue<Item>::DropAfterDequeue (item);
                m_managedCount++;
                s_totalManaged++;
                continue;
            }
        }
        return item;
    }
    return 0;
}

template <typename Item>
Ptr<Item>
PurgeSrpfQueue<Item>::Remove (void)
{
    return DoRemove (begin ());
}

template <typename Item>
Ptr<const Item>
PurgeSrpfQueue<Item>::Peek (void) const
{
    return DoPeek (begin ());
}

template <typename Item>
void
PurgeSrpfQueue<Item>::SetNodeId (int32_t node_id)
{
    m_nodeId = node_id;
}

template <typename Item>
uint64_t
PurgeSrpfQueue<Item>::GetPurgedCount (void) const
{
    return m_purgedCount;
}

template <typename Item>
uint64_t
PurgeSrpfQueue<Item>::GetTotalPurged (void)
{
    return s_totalPurged;
}

template <typename Item>
uint64_t
PurgeSrpfQueue<Item>::GetTotalManaged (void)
{
    return s_totalManaged;
}

extern template class PurgeSrpfQueue<Packet>;

} // namespace ns3

#endif /* PURGE_SRPF_QUEUE_H */
