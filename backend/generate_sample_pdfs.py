"""
generate_sample_pdfs.py
=======================
Generates 3 realistic synthetic legal PDFs for testing and demonstration.

Documents created:
  1. NDA-VendorX.pdf    — Non-Disclosure Agreement (Acme Corp ↔ Vendor X)
  2. SLA-ProviderY.pdf  — Service Level Agreement (Acme Corp ↔ Provider Y)
  3. IP-ContractorZ.pdf — IP Assignment Agreement (Acme Corp ↔ Contractor Z)

Run: python generate_sample_pdfs.py
Output: data/pdfs/*.pdf
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak
from reportlab.lib import colors

OUTPUT_DIR = Path("data/pdfs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STYLES = getSampleStyleSheet()

def _heading(text, size=14):
    return Paragraph(f"<b>{text}</b>", ParagraphStyle(
        "heading", fontSize=size, spaceAfter=8, spaceBefore=12,
        textColor=colors.HexColor("#1a1a2e")
    ))

def _body(text):
    return Paragraph(text, ParagraphStyle(
        "body", fontSize=10, leading=16, spaceAfter=6
    ))

def _clause(number, title, text):
    return [
        _heading(f"Clause {number}: {title}", size=11),
        _body(text),
        Spacer(1, 6),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# PDF 1: NDA — Non-Disclosure Agreement
# ─────────────────────────────────────────────────────────────────────────────

def create_nda():
    path = OUTPUT_DIR / "NDA-VendorX.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    story = [
        _heading("NON-DISCLOSURE AGREEMENT", size=18),
        _body("This Non-Disclosure Agreement (\"Agreement\") is entered into as of January 10, 2024, "
              "between <b>Acme Corporation Private Limited</b>, a company incorporated under the Companies Act, "
              "2013, having its registered office at 42, Business Park, Sector 18, Gurugram, Haryana 122015 "
              "(hereinafter referred to as \"Acme Corp\" or \"Disclosing Party\"), "
              "and <b>Vendor X Technologies Private Limited</b>, a company incorporated under the Companies Act, "
              "2013, having its registered office at 7, Tech Tower, Whitefield, Bengaluru, Karnataka 560066 "
              "(hereinafter referred to as \"Vendor X\" or \"Receiving Party\")."),
        Spacer(1, 12),
        _heading("RECITALS", size=12),
        _body("WHEREAS, Acme Corp and Vendor X (collectively, the \"Parties\") wish to explore a potential "
              "business relationship involving software development and IT infrastructure services; and"),
        _body("WHEREAS, in connection with the foregoing, it may be necessary for each Party to disclose "
              "certain confidential and proprietary information to the other Party;"),
        _body("NOW, THEREFORE, in consideration of the mutual covenants and agreements set forth herein, "
              "and for other good and valuable consideration, the receipt and sufficiency of which are hereby "
              "acknowledged, the Parties agree as follows:"),
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")),
        Spacer(1, 8),
        *_clause("1", "Definition of Confidential Information",
            "\"Confidential Information\" means any and all technical, commercial, financial, and business "
            "information disclosed by one Party (the \"Disclosing Party\") to the other Party (the \"Receiving Party\"), "
            "whether disclosed orally, in writing, by electronic means, or by inspection of tangible objects, "
            "including but not limited to: source code, algorithms, software architectures, trade secrets, "
            "customer lists, business plans, financial projections, pricing information, marketing strategies, "
            "product roadmaps, and any other information that is designated as \"Confidential\" or that reasonably "
            "should be understood to be confidential given the nature of the information and the circumstances "
            "of its disclosure."),
        *_clause("2", "Obligations of Receiving Party",
            "The Receiving Party agrees: (a) to hold the Confidential Information in strict confidence using "
            "at least the same degree of care it uses to protect its own confidential information, but in no "
            "event less than reasonable care; (b) not to disclose the Confidential Information to any third "
            "party without the prior written consent of the Disclosing Party; (c) to use the Confidential "
            "Information solely for the purpose of evaluating and pursuing the proposed business relationship "
            "between the Parties (the \"Purpose\"); and (d) to limit access to the Confidential Information "
            "to those employees, contractors, and advisors who have a need to know for the Purpose and who "
            "are bound by confidentiality obligations at least as stringent as those contained herein."),
        *_clause("3", "Duration of Confidentiality Obligations",
            "The obligations of confidentiality set forth in this Agreement shall commence on the Effective Date "
            "and shall continue for a period of <b>five (5) years</b> from the date of disclosure of the relevant "
            "Confidential Information, notwithstanding any earlier termination of this Agreement or of any "
            "business relationship between the Parties. For trade secrets, the obligations shall survive "
            "indefinitely to the extent permitted under applicable law."),
        PageBreak(),
        *_clause("4", "Exclusions from Confidentiality",
            "The obligations of confidentiality shall not apply to information that: (a) is or becomes publicly "
            "known through no breach of this Agreement by the Receiving Party; (b) was rightfully known by the "
            "Receiving Party prior to disclosure without any restriction on disclosure; (c) is independently "
            "developed by the Receiving Party without use of or reference to the Confidential Information; or "
            "(d) is required to be disclosed by applicable law, regulation, or court order, provided that the "
            "Receiving Party gives the Disclosing Party prompt written notice of such requirement and cooperates "
            "with the Disclosing Party in seeking a protective order."),
        *_clause("5", "Term and Termination / Notice Period",
            "This Agreement shall commence on the Effective Date and shall remain in effect for a period of "
            "twelve (12) months, unless earlier terminated. Either Party may terminate this Agreement by "
            "providing <b>thirty (30) days' written notice</b> to the other Party. Upon termination or expiration "
            "of this Agreement, the Receiving Party shall promptly return or destroy all Confidential Information "
            "in its possession, and shall provide written certification of such destruction within seven (7) "
            "business days of request. Termination shall not affect the survival of confidentiality obligations "
            "as set forth in Clause 3."),
        *_clause("6", "Governing Law and Jurisdiction",
            "This Agreement shall be governed by and construed in accordance with the laws of India, specifically "
            "the Indian Contract Act, 1872. Any disputes arising out of or in connection with this Agreement "
            "shall be subject to the exclusive jurisdiction of the courts located in New Delhi, India. The "
            "Parties agree to attempt in good faith to resolve any disputes through negotiation before resorting "
            "to litigation."),
        *_clause("7", "Limitation of Liability",
            "In no event shall either Party be liable to the other for any indirect, incidental, consequential, "
            "or punitive damages arising out of or related to this Agreement. The total aggregate liability of "
            "either Party under this Agreement, whether in contract, tort, or otherwise, shall not exceed "
            "<b>Indian Rupees Fifty Lakhs (₹50,00,000)</b>. This limitation of liability does not apply to "
            "breaches of confidentiality obligations, which shall be subject to equitable remedies including "
            "injunctive relief."),
        *_clause("8", "Entire Agreement",
            "This Agreement constitutes the entire agreement between the Parties with respect to the subject "
            "matter hereof and supersedes all prior and contemporaneous negotiations, agreements, and "
            "understandings between the Parties relating thereto. This Agreement may not be amended except "
            "by a written instrument signed by authorised representatives of both Parties."),
        Spacer(1, 24),
        _heading("SIGNATURES", size=12),
        _body("IN WITNESS WHEREOF, the Parties have executed this Non-Disclosure Agreement as of the date "
              "first written above."),
        Spacer(1, 16),
        _body("<b>Acme Corporation Private Limited</b>"),
        _body("Signature: _______________________"),
        _body("Name: Rajiv Mehta | Designation: Chief Executive Officer | Date: January 10, 2024"),
        Spacer(1, 16),
        _body("<b>Vendor X Technologies Private Limited</b>"),
        _body("Signature: _______________________"),
        _body("Name: Suresh Iyer | Designation: Managing Director | Date: January 10, 2024"),
    ]
    doc.build(story)
    print(f"Created: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PDF 2: SLA — Service Level Agreement
# ─────────────────────────────────────────────────────────────────────────────

def create_sla():
    path = OUTPUT_DIR / "SLA-ProviderY.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    story = [
        _heading("SERVICE LEVEL AGREEMENT", size=18),
        _body("This Service Level Agreement (\"SLA\" or \"Agreement\") is entered into as of March 1, 2024, "
              "between <b>Acme Corporation Private Limited</b> (\"Client\") and "
              "<b>Provider Y Cloud Services Limited</b>, a company having its principal place of business at "
              "Plot 15, Electronic City Phase II, Bengaluru, Karnataka 560100 (\"Service Provider\")."),
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")),
        Spacer(1, 8),
        *_clause("1", "Scope of Services",
            "The Service Provider agrees to provide the following managed cloud infrastructure services to the "
            "Client: (a) Cloud hosting and virtual machine management on ISO 27001-certified infrastructure; "
            "(b) 24×7 network monitoring and incident management; (c) Automated backup and disaster recovery; "
            "(d) Security patch management and vulnerability assessments; and (e) Monthly performance reporting "
            "and capacity planning. Services shall be rendered in accordance with the technical specifications "
            "set forth in Schedule A appended hereto."),
        *_clause("2", "Service Availability and Uptime Guarantee",
            "The Service Provider guarantees a monthly uptime of <b>99.9%</b> (the \"Uptime Guarantee\") for all "
            "production systems, calculated on a rolling calendar-month basis excluding scheduled maintenance "
            "windows. Uptime is calculated as: ((Total Minutes – Downtime Minutes) / Total Minutes) × 100%. "
            "Scheduled maintenance windows shall not exceed four (4) hours per month and shall be conducted "
            "between 02:00 and 06:00 IST on Sundays with a minimum of seventy-two (72) hours prior written "
            "notice to the Client."),
        *_clause("3", "Incident Response Times",
            "The Service Provider shall respond to and resolve incidents in accordance with the following "
            "priority matrix: "
            "Priority 1 (Critical — complete service outage): Initial response within <b>30 minutes</b>, "
            "resolution within <b>4 hours</b>. "
            "Priority 2 (High — significant degradation): Initial response within <b>2 hours</b>, "
            "resolution within <b>24 hours</b>. "
            "Priority 3 (Medium — partial impact): Initial response within <b>4 hours</b>, "
            "resolution within <b>72 hours</b>. "
            "Priority 4 (Low — minor issues): Initial response within <b>1 business day</b>, "
            "resolution within <b>10 business days</b>."),
        PageBreak(),
        *_clause("4", "Service Credits and Penalties",
            "In the event the Service Provider fails to meet the Uptime Guarantee in any given month, the "
            "Client shall be entitled to service credits calculated as follows: "
            "Uptime 99.0%–99.9%: Credit of 5% of monthly fee; "
            "Uptime 95.0%–99.0%: Credit of 15% of monthly fee; "
            "Uptime below 95.0%: Credit of 25% of monthly fee. "
            "Service credits are the Client's sole and exclusive remedy for failure to meet the Uptime "
            "Guarantee and must be claimed within thirty (30) days of the relevant month."),
        *_clause("5", "Payment Terms",
            "The Client shall pay all undisputed invoices within <b>thirty (30) days</b> of the invoice date "
            "(\"Net 30\"). The monthly service fee is Indian Rupees Eight Lakhs (₹8,00,000) per month, "
            "subject to annual revision with thirty (30) days' prior written notice. Late payments shall "
            "attract interest at the rate of 2% per month (24% per annum) compounded monthly on the "
            "outstanding amount. The Service Provider reserves the right to suspend services after "
            "sixty (60) days of non-payment."),
        *_clause("6", "Term and Termination / Notice Period",
            "This Agreement shall commence on the Effective Date and continue for an initial term of "
            "twenty-four (24) months, automatically renewing for successive twelve (12) month periods "
            "unless terminated. Either Party may terminate this Agreement without cause by providing "
            "<b>sixty (60) days' prior written notice</b> to the other Party. Immediate termination for cause "
            "is permitted upon: (a) material breach that remains uncured for fifteen (15) days after written "
            "notice; (b) insolvency or bankruptcy proceedings; or (c) regulatory action that prevents "
            "performance. Upon termination, the Service Provider shall provide data export assistance for "
            "ninety (90) days at no additional charge."),
        *_clause("7", "Limitation of Liability",
            "The aggregate liability of either Party under this Agreement, whether arising from contract, "
            "tort, negligence, or any other cause of action, shall be limited to the total fees paid by "
            "the Client to the Service Provider during the preceding twelve (12) months, subject to an "
            "absolute cap of <b>Indian Rupees Two Crore (₹2,00,00,000)</b>. Neither Party shall be liable "
            "for any indirect, consequential, incidental, punitive, or exemplary damages, including loss "
            "of revenue, loss of profits, or loss of data, even if advised of the possibility of such damages."),
        *_clause("8", "Data Protection and Security",
            "The Service Provider shall implement and maintain appropriate technical and organisational "
            "measures to protect the Client's data against unauthorised access, disclosure, alteration, "
            "and destruction. The Service Provider shall comply with all applicable data protection laws, "
            "including the Digital Personal Data Protection Act, 2023 (India). In the event of a data "
            "breach, the Service Provider shall notify the Client within twenty-four (24) hours of "
            "becoming aware of the breach."),
        Spacer(1, 16),
        _heading("SIGNATURES", size=12),
        _body("<b>Acme Corporation Private Limited</b>: Rajiv Mehta, CEO — Date: March 1, 2024"),
        _body("<b>Provider Y Cloud Services Limited</b>: Ananya Krishnamurthy, COO — Date: March 1, 2024"),
    ]
    doc.build(story)
    print(f"Created: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PDF 3: IP Assignment Agreement
# ─────────────────────────────────────────────────────────────────────────────

def create_ip_assignment():
    path = OUTPUT_DIR / "IP-ContractorZ.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    story = [
        _heading("INTELLECTUAL PROPERTY ASSIGNMENT AGREEMENT", size=16),
        _body("This Intellectual Property Assignment Agreement (\"Agreement\") is made and entered into as of "
              "February 15, 2024, between <b>Acme Corporation Private Limited</b> (\"Assignee\") and "
              "<b>Contractor Z — Priya Sharma</b>, an individual freelance software developer residing at "
              "204, Green Valley Apartments, Koramangala, Bengaluru 560034 (\"Assignor\")."),
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")),
        Spacer(1, 8),
        *_clause("1", "Background and Recitals",
            "WHEREAS, the Assignor has been engaged by the Assignee to develop a proprietary machine learning "
            "model for automated contract analysis (the \"Work\") pursuant to a consulting engagement "
            "commencing October 1, 2023; and WHEREAS, the Assignee desires to obtain full and exclusive "
            "ownership of all intellectual property rights in and to the Work and all related developments; "
            "NOW, THEREFORE, in consideration of the mutual covenants herein and for other good and valuable "
            "consideration, the Parties agree as follows:"),
        *_clause("2", "Assignment of Intellectual Property",
            "The Assignor hereby irrevocably and unconditionally assigns, transfers, and conveys to the Assignee "
            "all right, title, and interest, throughout the world, in and to: (a) all inventions, discoveries, "
            "concepts, and ideas, whether or not patentable, relating to the Work; (b) all copyrights, "
            "copyright registrations, and applications in the Work and all derivative works; (c) all trade "
            "secrets, know-how, and proprietary information embodied in the Work; and (d) all patents, "
            "patent applications, and all continuations, divisionals, and reissues thereof. The assignment "
            "is absolute and <b>full ownership of all intellectual property is hereby transferred to "
            "Acme Corporation Private Limited</b>, with no rights retained by the Assignor."),
        *_clause("3", "Consideration",
            "In full and complete consideration for the assignment of all intellectual property rights "
            "as described in Clause 2, the Assignee agrees to pay the Assignor a lump-sum consideration "
            "of <b>Indian Rupees Five Lakhs (₹5,00,000)</b>, payable within fifteen (15) business days "
            "of the execution of this Agreement. This consideration is inclusive of all taxes and levies "
            "applicable to the Assignor. The Assignee shall deduct TDS as applicable under Section 194J "
            "of the Income Tax Act, 1961, and issue Form 16A within the prescribed timelines."),
        PageBreak(),
        *_clause("4", "Moral Rights Waiver",
            "To the maximum extent permitted by applicable law, the Assignor hereby irrevocably waives "
            "all moral rights (including the right of attribution and the right of integrity) in and to "
            "the Work, and agrees not to assert any such rights against the Assignee, its licensees, "
            "or successors. This waiver is binding upon the Assignor's heirs, executors, and assigns."),
        *_clause("5", "Non-Compete and Non-Solicitation",
            "For a period of <b>two (2) years</b> from the date of this Agreement, the Assignor agrees not to: "
            "(a) directly or indirectly engage in, own, manage, operate, or provide services to any entity "
            "that competes with Acme Corp in the automated legal document analysis or contract intelligence "
            "space; (b) solicit or attempt to solicit any employee, consultant, or contractor of Acme Corp "
            "to leave their engagement; or (c) develop, commercialise, or license any work that is "
            "substantially similar to the Work assigned herein. This restriction applies within the "
            "territory of India and any country where Acme Corp has active commercial operations."),
        *_clause("6", "Term and Termination / Notice Period",
            "This Agreement shall take effect upon execution and shall remain in force perpetually with "
            "respect to the IP assignment. The non-compete and confidentiality obligations in Clauses 5 "
            "and 7 shall survive for the periods specified therein. Either Party may terminate any "
            "ongoing obligations (other than the IP assignment itself) by providing <b>fifteen (15) days' "
            "written notice</b> to the other Party. Termination of ongoing obligations shall not affect "
            "the irrevocability of the IP assignment."),
        *_clause("7", "Confidentiality",
            "The Assignor agrees to maintain in strict confidence all Confidential Information of the "
            "Assignee that comes into the Assignor's possession in connection with the performance of "
            "services or this Agreement. This obligation shall survive the termination of this Agreement "
            "for a period of three (3) years."),
        *_clause("8", "Limitation of Liability",
            "The aggregate liability of the Assignor to the Assignee for any breach of this Agreement "
            "shall not exceed <b>Indian Rupees Twenty-Five Lakhs (₹25,00,000)</b>. The Assignee's "
            "aggregate liability to the Assignor shall not exceed the Consideration paid under Clause 3. "
            "Neither Party shall be liable for indirect, incidental, or consequential damages."),
        *_clause("9", "Governing Law",
            "This Agreement shall be governed by the laws of India. Any disputes shall be resolved by "
            "binding arbitration under the Arbitration and Conciliation Act, 1996, with a sole arbitrator "
            "appointed by mutual consent of the Parties. The seat of arbitration shall be Bengaluru, "
            "Karnataka. The language of arbitration shall be English."),
        Spacer(1, 16),
        _heading("SIGNATURES", size=12),
        _body("<b>Acme Corporation Private Limited (Assignee)</b>: Rajiv Mehta, CEO — Date: February 15, 2024"),
        _body("<b>Contractor Z — Priya Sharma (Assignor)</b>: Priya Sharma — Date: February 15, 2024"),
    ]
    doc.build(story)
    print(f"Created: {path}")


if __name__ == "__main__":
    print("Generating sample legal PDFs...")
    create_nda()
    create_sla()
    create_ip_assignment()
    print(f"\nAll PDFs saved to: {OUTPUT_DIR.resolve()}")
