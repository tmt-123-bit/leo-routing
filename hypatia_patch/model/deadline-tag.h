/*
 * Deadline packet tag for the packet-management extension of Hypatia.
 *
 * Carries the end-to-end deadline information that the ISL
 * PurgeSrpfQueue needs: final destination node id (for hop-distance
 * chasing over the installed forwarding tables) and the absolute
 * deadline in nanoseconds since simulation start. Attached by the
 * deadline UDP sender at packet creation time.
 */

#ifndef DEADLINE_TAG_H
#define DEADLINE_TAG_H

#include "ns3/tag.h"
#include "ns3/tag-buffer.h"
#include "ns3/nstime.h"

namespace ns3 {

class DeadlineTag : public Tag
{
public:
    static TypeId GetTypeId (void);
    virtual TypeId GetInstanceTypeId (void) const;

    virtual uint32_t GetSerializedSize (void) const;
    virtual void Serialize (TagBuffer buffer) const;
    virtual void Deserialize (TagBuffer buffer);
    virtual void Print (std::ostream &os) const;

    DeadlineTag ();
    DeadlineTag (uint32_t dst_node_id, uint64_t deadline_ns,
                 uint32_t burst_id, uint32_t seq, uint64_t send_ns,
                 uint8_t traffic_class = 0);

    uint32_t GetDstNodeId () const;
    uint64_t GetDeadlineNs () const;
    uint32_t GetBurstId () const;
    uint32_t GetSeq () const;
    uint64_t GetSendNs () const;
    uint8_t GetTrafficClass () const;

private:
    uint32_t m_dst_node_id;
    uint64_t m_deadline_ns;
    uint32_t m_burst_id;
    uint32_t m_seq;
    uint64_t m_send_ns;
    uint8_t m_traffic_class;
};

} // namespace ns3

#endif /* DEADLINE_TAG_H */
