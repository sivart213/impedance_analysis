## Refined taxonomy with external relation anchoring

- **Element:** Atomic unit (R, C, L, W, etc.) per impedance.py definitions. No internal grouping. 
- **Path:** Series grouping of Elements only. No nested Networks; strictly primitive series.
- **Fork:** Parallel grouping of Elements only. No nested Networks; strictly primitive parallel.
- **Network:** Any arbitrary subcircuit that is not the full Model. Neutral container; may contain Elements, Paths, Forks, or other Networks. In impedance.py syntax, these are built via `s(...)` or `p(...)` constructs and nesting in the circuit string. 
- **Branch:** A Network whose external connection to its sibling Networks is parallel at the outer level (i.e., the parent relation is parallel).
- **Composite:** A Network whose external connection to its sibling Networks is series at the outer level (i.e., the parent relation is series).
- **Model:** The full circuit. Defined by the top-level circuit string (series separated by `-` and parallel via `p(...)`). 

---

## External relation rule

- **Definition anchor:** Branch/Composite classification is determined by the Network’s relation to its siblings in the parent context, not by the Network’s internal shape.
  - **Branch:** Parent combines this Network in parallel with at least one other Network.
  - **Composite:** Parent combines this Network in series with at least one other Network.
- **Neutral case:** If a Network has no siblings (e.g., the entire Model is a single Network), leave it as Network without forcing Branch/Composite.

---

## Grammar breakdown

```ebnf
(* The full Circuit *)
Model = Component | Circuit ;

(* Any valid collection of components connected in series or parallel *)
Circuit = Series | Parallel ;

(* Groupings of circuit components *)
(* Series: 2 or more valid Composites separated by "-" *)
Series = Composite , "-" , Composite , { "-" , Composite } ;

(* Items in series may be components or parallel groups *)
Composite = Component | Parallel ;

(* Parallel: 2 or valid Branches separated by "," and wrapped in p(...) *)
Parallel = "p(" , Branch , "," , Branch , { "," , Branch } , ")" ;

(* Items in series may be components or series groups *)
Branch = Component | Series ;

(* Special primitive groupings of base elements only *)
Path = Component , "-" , Component, { "-" , Component } ;
Fork = "p(" , Component , "," , Component , { "," , Component } , ")" ;

(* Identifiers used when referring to fitting parameters *)
(* These extend the Component name with an extra underscore + index, for multi-input components *)
(* They never appear in the circuit string, only in parameter dictionaries or constants *)
Parameter   = Component, [ "_" , Digit , { Digit } ] ;

(* Identifiers used in the circuit string itself *)
Component   = Element , [ "_" ] , Digit , { Digit } ;

(* Element names available in impedance.py *)
(* Each must be suffixed with an index to distinguish multiple instances *)
Element  = "R" | "C" | "L" | "W" | "Wo" | "Ws"
             | "G" | "Gs" | "K" | "La" | "T" | "TLMQ"
             | "CPE" | "ICPE" | "Zarc" ;

Digit  = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
```

## Classification on a concrete example

```text
Model: R0 - p(R1, C1) - p(R2 - Wo2, C2)
└── R0 - p(R1, C1) - p(R2 - Wo2, C2) -> Series:
    ├── "R0" -> Composite -> Component
    │   ├── "R" -> Element
    │   └── "0" -> Digit
    ├── "-" 
    ├── "p(R1, C1)" -> Composite -> Parallel
    │   ├── "p("
    │   ├── "R1" -> Branch -> Component
    │   │   ├── "R" -> Element
    │   │   └── "1" -> Digit
    │   ├── ","
    │   ├── "C1" -> Branch -> Component
    │   │   ├── "C" -> Element
    │   │   └── "1" -> Digit
    │   └── ")"
    ├── "-"
    └── "p(R2 - Wo2, C2)" -> Composite -> Parallel
        ├── "p("
        ├── "R2 - Wo2" -> Branch -> Series
        │   ├── "R2" -> Composite -> Component
        │   │   ├── "R" -> Element
        │   │   └── "2" -> Digit
        │   ├── "-" 
        │   └── "Wo1" -> Composite -> Component
        │       ├── "Wo" -> Element
        │       └── "2" -> Digit
        ├── ","
        ├── "C2" -> Branch -> Component
        │   ├── "C" -> Element
        │   └── "2" -> Digit
        └── ")"
        

```
---

## Edge cases and disambiguation

- **Single-element Networks:** An Element can be treated as a degenerate Network for uniform handling; its Branch/Composite status still depends on the parent relation.
- **Nested parallel-in-series (and vice versa):** Internal nesting does not affect Branch/Composite labeling; only the parent relation does. Use Element/Path/Fork to describe the primitive internal shape, and Branch/Composite for the external role.
- **Multiple siblings mixed:** If a parent has both series and parallel at different nesting levels, classify each child Network by the immediate parent operator:
  - Parent `p(...)` → children are Branches.
  - Parent series (`s(...)` or `-`) → children are Composites. 
- **Root-only Network:** If the Model consists of a single Network (no `-` or `p(...)` at top), it is considered an Element, Network, and Model.

---

## Practical parsing rules for implementation

- **Rule 1:** Parse to a tree where nodes are operators: Series (from `-` or `s(...)`) and Parallel (`p(...)`). Leaves are Elements. 
- **Rule 2:** Label internal primitives:
  - Node Parallel with only Element children → Fork.
  - Node Series with only Element children → Path.
- **Rule 3:** Label external roles:
  - For any child Network of a Parallel node → Branch.
  - For any child Network of a Series node → Composite.
- **Rule 4:** Preserve neutral Network label when a node has no siblings at its parent.

---

## Menu wording suggestions

- **Create Path:** Group selected Elements in series.
- **Create Fork:** Group selected Elements in parallel.
- **Group as Composite:** Treat selection as a Network in series with siblings.
- **Group as Branch:** Treat selection as a Network in parallel with siblings.
- **Promote to Network:** Name and reuse a subcircuit.
- **Insert Network:** Add a reusable subcircuit into the Model.

This keeps Path/Fork for primitive clarity and Branch/Composite for external structural roles, aligned with impedance.py’s `-` and `p(...)` grammar.

[impedance.py getting started](https://impedancepy.readthedocs.io/en/latest/getting-started.html)
[impedance.py circuit elements](https://impedancepy.readthedocs.io/en/latest/circuit-elements.html#)  
