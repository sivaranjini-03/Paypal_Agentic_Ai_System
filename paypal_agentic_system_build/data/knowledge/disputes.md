# Disputes and claims

A dispute is opened by a buyer who is unhappy with a transaction. Each dispute
has a `dispute_id` and moves through a lifecycle that constrains which actions
are legal.

## Lifecycle

1. `OPEN` - the buyer raised the issue; the parties can exchange messages.
2. `WAITING_FOR_SELLER_RESPONSE` - the seller must act, usually by providing
   evidence or making an offer.
3. `WAITING_FOR_BUYER_RESPONSE` - the buyer must act.
4. `UNDER_REVIEW` - the platform is adjudicating.
5. `RESOLVED` - the dispute is closed.

A dispute becomes a claim when it is escalated. Escalation is only possible
from a dispute that is still open and within the eligible window.

## Common actions and their preconditions

- Accept claim: the seller concedes; the buyer is refunded. Only valid while the
  dispute is still actionable.
- Make offer to resolve: the seller proposes a refund, replacement, or partial
  refund. The buyer may accept or deny it.
- Provide evidence: upload proof such as tracking or proof of delivery. Usually
  only valid while the dispute is waiting for the seller.
- Send message: exchange notes with the other party; valid in most open states.
- Acknowledge returned item: confirm receipt of a returned product.
- Appeal: available after an unfavourable resolution, within the appeal window.

Acting out of order typically fails with an issue code describing the invalid
state rather than a network error. These are business errors and should not be
retried blindly.

## Finding disputes

Listing disputes supports filtering by state, time range and related
transaction, and is how to locate a `dispute_id` when the user describes a
customer or an order rather than an identifier.
