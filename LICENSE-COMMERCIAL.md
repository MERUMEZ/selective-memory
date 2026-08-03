# Commercial licence for selectivemem — DRAFT

> **This is a working draft, not a contract.** A lawyer must review it
> before the first deal: the wording on liability, governing law and
> payment depends on both parties' jurisdictions, and a mistake there
> costs more than the licence itself. What is fixed here is the substance
> of what the parties agree on — so that a lawyer has something to
> correct rather than something to compose from nothing.

---

## 1. Parties and subject

**Licensor:** MERUMEZ, selectivemem@gmail.com
*(to be replaced with a full legal name or company details)*

**Licensee:** ______________________

**Subject:** the software library `selectivemem` (the Program), whose
source is published at https://github.com/MERUMEZ/mindnumbness

The Program is publicly distributed under AGPL-3.0. This licence is
granted **in addition to** and **instead of** AGPL: having received it,
the Licensee is not required to comply with AGPL in respect of the
Program.

## 2. What is permitted

The Licensor grants a non-exclusive, non-transferable right to:

1. use the Program within the Licensee's Product **without disclosing
   the Product's source code**;
2. distribute the Program within the Product in object or executable
   form;
3. provide access to the Program's functionality **over a network**
   without the obligations of AGPL §13 (conveying source to users);
4. modify the Program for the needs of the Product.

**Product:** ______________________
*(specific name; see §4 on scope)*

## 3. What is not permitted

1. Distributing the Program **as a standalone product** — that is,
   selling or conveying it on its own, outside a Product.
2. Transferring the rights under this licence to third parties,
   including on a merger or sale of the business, without the Licensor's
   written consent.
3. Removing attribution notices from the source files.
4. Using the name `selectivemem` or the Licensor's name to promote the
   Product without separate consent.

## 4. Scope

Choose **one**; the price depends on it.

| Option | What is covered |
|---|---|
| **One product** | a single named product, including its updates and re-releases |
| **One game / title** | one game, including expansions and ports to other platforms |
| **Organisation** | all of the Licensee's products, with no limit on their number |

Chosen: ______________________

## 5. Term and versions

The licence is **perpetual** in respect of versions of the Program
released within ______ (typically 12) months of payment.

That means: a Product built on such a version may be distributed and
supported indefinitely. Versions released later require renewal.

Renewal for the next period: ______ % of the original sum.

## 6. Fee

Amount: ______________________
Payment terms: ______________________

*(Reference ranges are in [COMMERCIAL.md](COMMERCIAL.md). The final sum
is determined by scope and scale.)*

## 7. Support

By default the licence **does not include** support or custom work.

If the parties agree otherwise, this section records the response time,
the channel and the amount of work. Custom integration is contracted
separately.

## 8. Warranties and liability

The Program is provided "as is". The Licensor warrants only that it is
entitled to grant this licence, and gives no warranty of fitness for the
Licensee's particular purposes.

The Licensor's liability is limited to the sum actually paid under this
licence.

> To the lawyer: this section is the first that will need aligning with
> the governing law. Limiting liability to the contract sum is not
> recognised in every jurisdiction.

## 9. What the Licensee should know beforehand

This section is optional, and it is here deliberately. Better for an
inconvenient truth to be said before the deal than to surface after it.

- **The Program's measured advantage is telling apart memories that look
  alike.** Under 200 near-duplicates it puts the right one first 83% of
  the time against 50% for weight-and-recency scoring.
- **On a store of unrelated topics that advantage disappears entirely**
  (76.0% either way on LongMemEval). If your data is topically diverse,
  an ordinary vector store is no worse and this Program is not worth
  paying for.
- **Selective writing costs recall.** On the external LongMemEval
  benchmark, 500 questions at stock settings: R@5 84.0% while storing
  24.7% of the turns, against 93.2% for the same engine with the write
  filter removed. Nine points for three quarters of the storage.
- **A prior claim of "+40 pp selective retention" has been withdrawn.**
  It measured deletion rather than preference; removing that deletion
  raised recall by 18.6 points. The audit records this in full. It is
  named here because a buyer who reads older material should hear it
  from us first.
- **A mismatched encoder CORRUPTS memory, it does not merely search
  badly.** Russian vectors over English text score unrelated sentences at
  0.808, above the supersession threshold: 3080 spurious weakenings
  across 79 writes. Pass an encoder for your language.
- **Preference questions are the weak row:** R@5 49.6% against 76.8% for
  the unfiltered engine.
- **Search is linear in the number of nodes.** Roughly 14 ms per 1000
  nodes on an ordinary machine once the vector cache is warm.

## 10. Governing law and disputes

______________________

> To the lawyer: depends on both parties' jurisdictions. For deals with
> foreign counterparties, the payment mechanics need separate thought.

---

**Licensor:** ____________________  date ____________

**Licensee:** ____________________  date ____________

---

Русская версия: [LICENSE-COMMERCIAL.ru.md](LICENSE-COMMERCIAL.ru.md).
