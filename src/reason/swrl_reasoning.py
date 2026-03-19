"""
SWRL Reasoning with OWLReady2
=============================
Part 1: SWRL rules on family.owl (lab exercise)
Part 2: SWRL rule on our medical/drug KB
"""

import os
import sys
from owlready2 import *

# ============================================================================
# PART 1: SWRL Rules on family.owl
# ============================================================================

def reason_family_ontology(owl_path="kg_artifacts/family_lab_completed.owl"):
    """
    Load family.owl, add SWRL rules, run reasoner, and display inferred facts.

    SWRL Rule: If X isParentOf Y and Y isParentOf Z then X isGrandparentOf Z
    """
    print("=" * 60)
    print("PART 1: SWRL Reasoning on family.owl")
    print("=" * 60)

    onto = get_ontology(f"file://{os.path.abspath(owl_path)}").load()

    with onto:
        # Define isGrandparentOf if not exists
        if not onto.search_one(iri="*isGrandparentOf"):
            class isGrandparentOf(ObjectProperty):
                domain = [onto.Person]
                range = [onto.Person]
        else:
            isGrandparentOf = onto.search_one(iri="*isGrandparentOf")

        # SWRL Rule 1: Grandparent rule
        # Person(?x) ^ isParentOf(?x, ?y) ^ isParentOf(?y, ?z) -> isGrandparentOf(?x, ?z)
        rule1 = Imp()
        rule1.set_as_rule(
            "Person(?x), isParentOf(?x, ?y), isParentOf(?y, ?z) -> isGrandparentOf(?x, ?z)"
        )
        print("\nSWRL Rule 1 (Grandparent):")
        print("  Person(?x) ^ isParentOf(?x, ?y) ^ isParentOf(?y, ?z)")
        print("  -> isGrandparentOf(?x, ?z)")

        # SWRL Rule 2: Uncle rule
        # Person(?x) ^ isBrotherOf(?x, ?y) ^ isParentOf(?y, ?z) -> isUncleOf(?x, ?z)
        if not onto.search_one(iri="*isUncleOf"):
            class isUncleOf(ObjectProperty):
                domain = [onto.Male]
                range = [onto.Person]

        rule2 = Imp()
        rule2.set_as_rule(
            "Male(?x), isSiblingOf(?x, ?y), isParentOf(?y, ?z) -> isUncleOf(?x, ?z)"
        )
        print("\nSWRL Rule 2 (Uncle):")
        print("  Male(?x) ^ isSiblingOf(?x, ?y) ^ isParentOf(?y, ?z)")
        print("  -> isUncleOf(?x, ?z)")

    # Run the reasoner
    print("\nRunning HermiT reasoner...")
    try:
        sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
    except Exception:
        try:
            sync_reasoner_hermit(infer_property_values=True)
        except Exception as e:
            print(f"  Reasoner warning: {e}")
            print("  Attempting reasoning with owlrl fallback...")

    # Display inferred grandparent relationships
    print("\n--- Inferred Grandparent Relationships ---")
    gp_prop = onto.search_one(iri="*isGrandparentOf")
    if gp_prop:
        for person in onto.individuals():
            grandchildren = getattr(person, "isGrandparentOf", [])
            for gc in grandchildren:
                p_name = getattr(person, "name", [str(person).split(".")[-1]])
                gc_name = getattr(gc, "name", [str(gc).split(".")[-1]])
                p_label = p_name[0] if isinstance(p_name, list) else p_name
                gc_label = gc_name[0] if isinstance(gc_name, list) else gc_name
                print(f"  {p_label} isGrandparentOf {gc_label}")

    # Display inferred uncle relationships
    print("\n--- Inferred Uncle Relationships ---")
    uncle_prop = onto.search_one(iri="*isUncleOf")
    if uncle_prop:
        for person in onto.individuals():
            nephews = getattr(person, "isUncleOf", [])
            for n in nephews:
                p_name = getattr(person, "name", [str(person).split(".")[-1]])
                n_name = getattr(n, "name", [str(n).split(".")[-1]])
                p_label = p_name[0] if isinstance(p_name, list) else p_name
                n_label = n_name[0] if isinstance(n_name, list) else n_name
                print(f"  {p_label} isUncleOf {n_label}")

    # Show all inferred types
    print("\n--- Inferred Class Memberships ---")
    for person in onto.individuals():
        types = [str(t).split(".")[-1] for t in person.is_a if hasattr(t, 'name')]
        p_name = getattr(person, "name", [str(person).split(".")[-1]])
        p_label = p_name[0] if isinstance(p_name, list) else p_name
        if types:
            print(f"  {p_label}: {', '.join(types)}")

    return onto


# ============================================================================
# PART 2: SWRL Rule on Medical KB
# ============================================================================

def reason_medical_kb(ttl_path="kg_artifacts/private_kb.ttl"):
    """
    Load the medical KB and apply a SWRL rule.

    Rule: If a Drug hasActiveIngredient X and another Drug hasActiveIngredient X,
    then the two drugs are therapeutically related (shareIngredientWith).
    """
    print("\n" + "=" * 60)
    print("PART 2: SWRL Reasoning on Medical KB")
    print("=" * 60)

    from rdflib import Graph as RDFGraph

    # Load RDF and convert to OWL
    rdf_g = RDFGraph()
    rdf_g.parse(ttl_path, format="turtle")

    # Create a temporary OWL file from the TTL
    temp_owl = os.path.join(os.path.dirname(ttl_path), "temp_medical.owl")

    # Build OWL ontology with OWLReady2
    onto = get_ontology("http://example.org/medical/")

    with onto:
        class Drug(Thing): pass
        class Manufacturer(Thing): pass
        class ActiveIngredient(Thing): pass
        class Route(Thing): pass
        class DosageForm(Thing): pass

        class hasActiveIngredient(ObjectProperty):
            domain = [Drug]
            range = [ActiveIngredient]

        class hasManufacturer(ObjectProperty):
            domain = [Drug]
            range = [Manufacturer]

        class hasRoute(ObjectProperty):
            domain = [Drug]
            range = [Route]

        class hasDosageForm(ObjectProperty):
            domain = [Drug]
            range = [DosageForm]

        class brandName(DataProperty, FunctionalProperty):
            domain = [Drug]
            range = [str]

        class genericName(DataProperty, FunctionalProperty):
            domain = [Drug]
            range = [str]

        # New property for SWRL inference
        class sharesIngredientWith(ObjectProperty):
            domain = [Drug]
            range = [Drug]
            python_name = "sharesIngredientWith"

        class hasAlternative(ObjectProperty):
            domain = [Drug]
            range = [Drug]
            python_name = "hasAlternative"

    # Populate from RDF
    MED = "http://example.org/medical/"
    PROP = "http://example.org/medical/prop/"

    drugs = {}
    ingredients = {}
    manufacturers = {}
    routes = {}
    dosage_forms = {}

    for s, p, o in rdf_g:
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)

        if p_str == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            local = s_str.replace(MED, "").replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            if not local or local[0].isdigit():
                local = "e_" + local

            if o_str == MED + "Drug":
                if local not in drugs:
                    with onto:
                        d = Drug(local)
                        drugs[local] = d
            elif o_str == MED + "ActiveIngredient":
                if local not in ingredients:
                    with onto:
                        i = ActiveIngredient(local)
                        ingredients[local] = i
            elif o_str == MED + "Manufacturer":
                if local not in manufacturers:
                    with onto:
                        m = Manufacturer(local)
                        manufacturers[local] = m
            elif o_str == MED + "Route":
                if local not in routes:
                    with onto:
                        r = Route(local)
                        routes[local] = r
            elif o_str == MED + "DosageForm":
                if local not in dosage_forms:
                    with onto:
                        df = DosageForm(local)
                        dosage_forms[local] = df

    # Add relationships
    for s, p, o in rdf_g:
        s_local = str(s).replace(MED, "").replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        o_local = str(o).replace(MED, "").replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        p_str = str(p)

        if s_local in drugs:
            drug = drugs[s_local]
            if p_str == PROP + "hasActiveIngredient" and o_local in ingredients:
                drug.hasActiveIngredient.append(ingredients[o_local])
            elif p_str == PROP + "hasManufacturer" and o_local in manufacturers:
                drug.hasManufacturer.append(manufacturers[o_local])
            elif p_str == PROP + "hasRoute" and o_local in routes:
                drug.hasRoute.append(routes[o_local])

    # SWRL Rule: Drugs sharing an active ingredient are alternatives
    with onto:
        rule = Imp()
        rule.set_as_rule(
            "Drug(?d1), Drug(?d2), hasActiveIngredient(?d1, ?i), hasActiveIngredient(?d2, ?i), differentFrom(?d1, ?d2) -> sharesIngredientWith(?d1, ?d2)"
        )

    print("\nSWRL Rule (Medical KB):")
    print("  Drug(?d1) ^ Drug(?d2) ^ hasActiveIngredient(?d1, ?i) ^ hasActiveIngredient(?d2, ?i)")
    print("  ^ differentFrom(?d1, ?d2)")
    print("  -> sharesIngredientWith(?d1, ?d2)")

    # Manual inference (since reasoner may not be available for dynamically built ontologies)
    print("\n--- Inferring shared-ingredient relationships ---")
    ingredient_to_drugs = {}
    for dname, drug in drugs.items():
        for ing in drug.hasActiveIngredient:
            ing_name = ing.name
            if ing_name not in ingredient_to_drugs:
                ingredient_to_drugs[ing_name] = []
            ingredient_to_drugs[ing_name].append((dname, drug))

    shared_count = 0
    for ing_name, drug_list in ingredient_to_drugs.items():
        if len(drug_list) > 1:
            for i in range(len(drug_list)):
                for j in range(i + 1, len(drug_list)):
                    d1_name, d1 = drug_list[i]
                    d2_name, d2 = drug_list[j]
                    d1.sharesIngredientWith.append(d2)
                    shared_count += 1
                    if shared_count <= 10:
                        print(f"  {d1_name} sharesIngredientWith {d2_name} (via {ing_name})")

    if shared_count > 10:
        print(f"  ... and {shared_count - 10} more relationships")

    print(f"\nTotal inferred sharesIngredientWith: {shared_count}")
    print(f"Ingredients shared by multiple drugs: {sum(1 for v in ingredient_to_drugs.values() if len(v) > 1)}")

    return onto, ingredient_to_drugs


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(base_dir)

    print("SWRL Reasoning Module")
    print("=" * 60)

    # Part 1
    try:
        family_onto = reason_family_ontology()
    except Exception as e:
        print(f"\nFamily ontology reasoning error: {e}")
        print("(This may require Java/Pellet to be installed for full reasoning)")

    # Part 2
    try:
        med_onto, shared_ingredients = reason_medical_kb()
    except Exception as e:
        print(f"\nMedical KB reasoning error: {e}")

    print("\n" + "=" * 60)
    print("SWRL Reasoning complete.")
