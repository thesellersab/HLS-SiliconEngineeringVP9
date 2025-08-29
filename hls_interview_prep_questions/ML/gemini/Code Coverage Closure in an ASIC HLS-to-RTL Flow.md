

# **Code Coverage Closure in an ASIC HLS-to-RTL Flow: A Comprehensive Guide Using Siemens Catapult HLS, SCVerify, and Questa**


![](https://github.com/thesellersab/icons_image_emblems_signs/blob/main/image.png)
![](https://github.com/thesellersab/icons_image_emblems_signs/blob/main/edshaz_7.png)



### **1\. Executive Summary and the Coverage Sign-off Checklist**

The evolution of Application-Specific Integrated Circuit (ASIC) design has introduced a critical paradigm shift: High-Level Synthesis (HLS). By enabling designers to specify hardware behavior in high-level languages like C, C++, and SystemC, HLS significantly accelerates the design and verification process, leading to substantial productivity gains and faster time-to-market. A core component of this advanced workflow is the ability to plan, collect, and analyze code coverage metrics seamlessly across abstraction levels. This report outlines a robust, end-to-end methodology for achieving auditable code coverage sign-off for real silicon designs using a cohesive Siemens EDA toolchain, including Catapult HLS, SCVerify, and Questa. The approach focuses on front-loading verification and debug to the algorithmic C++ level, where simulation is orders of magnitude faster than at the Register-Transfer Level (RTL), and then leveraging automation to ensure that this high-level verification effort is fully traceable to the final gate-level netlist. The objective is to demonstrate a reproducible flow that provides high confidence in design quality by unifying coverage data from source to synthesized hardware.

The following checklist provides a quick-reference guide for a successful coverage closure process:

* Define a comprehensive, requirements-driven test plan that links high-level features to specific coverage metrics.  
* Achieve a target percentage of code coverage (e.g., 95% line and branch) at the C/C++/SystemC level using a software-centric tool (llvm-cov).  
* Synthesize the HLS source to RTL using Catapult, ensuring source-code traceability is enabled.  
* Execute a full RTL simulation regression, collecting coverage data (.ucdb) from all tests.  
* Merge all individual coverage databases into a single, unified database (regression.ucdb).  
* Triage all remaining coverage holes, categorizing them as stimulus gaps, checker errors, or unreachable code.  
* Formally analyze and prove unreachability for a critical subset of remaining holes, and generate auditable waivers with a clear rationale.  
* Meet or exceed all defined coverage thresholds on all design units, with all waivers documented and approved.  
* Run a small, targeted set of tests at the gate-level (GLS) to collect performance-critical metrics like toggle coverage and FSM coverage.  
* Archive all verification artifacts, including the final unified coverage database, logs, reports, and waiver ledgers, for long-term traceability and future audits.

### **2\. Foundational Concepts: Verification Taxonomy and Strategy**

#### **The Imperative of Code Coverage**

In hardware verification, two primary metrics guide the verification process: code coverage and functional coverage. Code coverage is an automated, implementation-centric metric that measures which parts of the design's code have been exercised during simulation. This includes metrics like line, statement, and branch execution. Conversely, functional coverage is a user-defined, specification-centric metric that measures how much of the design's intended functionality has been verified. A well-known example illustrates the critical distinction: a design specification might require three features (A, B, and C), but the RTL only implements features A and B. A verification engineer could achieve 100% code coverage by thoroughly testing only features A and B, yet a massive functional hole (feature C) would remain in the design.1 This demonstrates that while code coverage is a necessary metric—it is impossible to verify unexecuted code—it is an insufficient measure of design correctness. Both code and functional coverage must be used in tandem to ensure a high-quality design.1

Within an HLS-driven flow, a unique opportunity arises. The C++ or SystemC source code serves as the golden model, representing the design's functional and algorithmic intent. When this C++ source code is rigorously tested to a high code coverage standard, it provides strong assurance that the underlying algorithm is well-exercised before it is even synthesized to RTL. This approach shifts the verification burden to a higher, faster-simulating abstraction level, where bugs are cheaper and easier to fix. By combining high C++ code coverage with a formal equivalence check between the C++ and the generated RTL, a verification team can be highly confident that the silicon implementation accurately reflects the verified algorithmic intent. In this context, C++ code coverage effectively serves as a proxy for functional coverage of the block's behavior, since the C++ model is considered the specification itself.

#### **A Unified Code Coverage Model**

A robust verification methodology requires a granular understanding of code coverage metrics to effectively measure test completeness and identify verification gaps.

* **Statement/Line Coverage:** This is the most basic metric, reporting whether each executable line of code has been hit at least once. It indicates which parts of the design's logic have been "touched" by the testbench.3  
* **Branch/Decision Coverage:** This metric goes deeper than statement coverage by checking if all possible outcomes of a decision point (e.g., if-else statements, case statements) have been taken. A simple if statement with no else implicitly contains a "false" branch that must be covered.3  
* **Condition Coverage:** This metric evaluates each boolean sub-expression within a larger conditional expression. For example, in an expression A && B, condition coverage ensures that A has evaluated to both true and false, and B has also evaluated to true and false.3  
* **Expression Coverage (Focused Expression Coverage \- FEC):** This is an extension of condition coverage that ensures all possible combinations of inputs to a boolean expression have been hit.4  
* **FSM State and Transition Coverage:** For designs with control logic, FSM coverage is essential. It tracks whether all states of a Finite State Machine have been entered and all legal transitions between states have been taken.3  
* **Toggle Coverage:** At a fundamental level, toggle coverage tracks if every bit of every signal has transitioned from 0 to 1 and 1 to 0\.3 This metric is particularly useful for identifying floating nodes or dead logic in the gate-level netlist, as a signal that never toggles may indicate a serious design or testbench issue. It is also an important metric for power analysis.6

#### **Assertion-Driven Coverage**

SystemVerilog Assertions (SVA) provide a powerful mechanism to monitor design behavior directly within the RTL. There are three key types of SVA statements: assert, assume, and cover.7 While

assert checks for violations of design intent and assume constrains inputs for formal tools, cover property is specifically designed for collecting functional coverage. The utility of cover property is its ability to monitor complex temporal sequences that are difficult to track with traditional code coverage or even procedural functional coverage models.9 For example, a

cover property can be written to check if a handshake protocol completes within a specific number of clock cycles (a |=\> \#\#\[1:5\] b;).10 This level of detail provides an indispensable metric for coverage closure. A key distinction between

assert and cover is that an assert failure typically invalidates the entire test run, as the design has entered an illegal state. In contrast, cover property simply reports a "hit" or "miss" without failing the simulation, making it a more flexible tool for measuring verification progress.10

### **3\. HLS-Level Coverage (C/C++/SystemC): The Algorithmic Foundation**

Verifying the HLS C++ source code is the most efficient part of the verification flow. Simulation at this level can be hundreds of times faster than RTL simulation, allowing for extensive design space exploration and bug hunting before the time-consuming synthesis step.11

#### **Toolchain Setup and C-Level Instrumentation**

The recommended approach for collecting C-level code coverage is to use the llvm-cov toolchain. While gcov is a mature alternative, llvm-cov provides superior support for modern C++ constructs, including templates and complex data types, which are heavily used in HLS libraries such as ac\_datatypes and Matchlib.13

A standard C-level code coverage flow involves three main steps:

1. **Compilation:** The source code and testbench are compiled with specific instrumentation flags. For clang++, the command is:  
   Bash  
   clang++ \-g \-O0 \-fprofile-instr-generate \-fcoverage-mapping \<src.cpp\> \<tb.cpp\> \-o \<test\_exe\>

   The \-fprofile-instr-generate flag adds instrumentation to the executable, while \-fcoverage-mapping links the executable back to the source code for reporting.13  
2. **Execution:** The instrumented executable is run. This generates a raw coverage data file (.profraw).13

./\<test\_exe\>  
\`\`\`

3. **Reporting:** The raw data is processed and formatted into a human-readable report. This is typically a two-step process: merging the raw data into a profile data file, and then generating the report itself.  
   Bash  
   llvm-profdata merge \-o default.profdata \<test\_exe\>.profraw  
   llvm-cov show./\<test\_exe\> \-instr-profile=default.profdata

   This process can generate line-by-line coverage reports in the terminal or in a more detailed HTML format.13

While a generic C++ coverage tool like llvm-cov is valuable for initial test development, it may not perfectly capture the nuances of a synthesizable C++ dialect. For example, it might not handle the effects of HLS directives like function inlining or loop unrolling, which change the final hardware implementation. For this reason, it is recommended to use C-level coverage for architectural exploration and test completeness, but to rely on Catapult’s native HLS-aware coverage for the final C-level metrics. Catapult's internal coverage tools are designed to understand how HLS directives impact the resulting hardware and can report statement, branch, and expression coverage on the C++ source in a way that is consistent with the final RTL.5

#### **Catapult HLS-Aware Coverage Integration**

The Catapult HLS flow includes an integrated coverage solution that directly outputs coverage data to the Questa Unified Coverage Database (UCDB).5 This is a crucial feature, as it allows for a unified view of verification metrics across different abstraction levels. This native integration enables the HLS-aware coverage data, which is generated from the fast C++ simulations, to be seamlessly merged with coverage data collected from slower RTL regressions.5 This unified approach provides a single, authoritative source for all coverage metrics, streamlining the closure process.

### **4\. The Catapult HLS to RTL Verification Flow**

After the C++ design has been thoroughly verified at the high level, the next step is to ensure that the generated RTL is functionally identical. Catapult’s automated verification flow, SCVerify, provides a push-button solution for this critical task.

#### **SCVerify: The Automated Verification Bridge**

The SCVerify flow automates the co-simulation of the C++ and RTL models.15 It takes the original C++ testbench and automatically generates a SystemC test infrastructure that connects to the generated RTL.12 This allows the same high-level test vectors that were used to verify the C++ algorithm to be reused to drive the RTL, and the results are automatically compared against the C++ reference model. This push-button flow provides a high degree of confidence in the functional equivalence of the C++ and RTL, reducing the time and effort required for unit-level verification. The SCVerify flow provides an ideal solution for sanity-checking the synthesized RTL before handing it off to the broader RTL verification team.12

#### **RTL Verification Environment: A Hybrid Approach**

For a complete verification of a complex IP block, especially one that is to be integrated into a larger System-on-Chip (SoC), a more comprehensive testbench is required. A Universal Verification Methodology (UVM) or plain SystemVerilog (SV) environment is the industry standard for this task. The challenge is to leverage the fast C++ model within this slower RTL environment. The solution lies in using the SystemVerilog Direct Programming Interface (DPI-C).17

The C++ reference model, which was verified at the high level, can be compiled into a shared library. The SV testbench can then call functions from this C++ model using DPI-C to generate stimulus, drive transactions, and check results.19 This hybrid approach preserves the investment in the high-level code, allowing it to serve as the golden reference model for the entire downstream verification process. A simple UVM-lite testbench can be structured with a driver, a monitor, and a scoreboard. The monitor captures transactions from the RTL Design Under Test (DUT), and the scoreboard compares them against the results generated by the C++ reference model called via DPI-C. This robust architecture ensures the functional correctness of the generated RTL.18

#### **C++ to RTL Traceability**

A crucial part of debugging and analysis is the ability to correlate the generated RTL back to its original C++ source code. Catapult HLS provides internal reports and cross-probing features that allow a user to trace a specific line of C++ code to the corresponding RTL signals and state machines.21 This capability is invaluable for debugging functional mismatches or for understanding why a specific coverage hole in the RTL exists. By visually mapping the logic, a verification engineer can quickly determine if the issue is a design bug or a testbench deficiency.22

### **5\. RTL and Gate-Level Coverage with Questa**

The final layer of code coverage is collected during RTL and gate-level simulation using Questa. This is the stage where the comprehensive metrics—statement, branch, condition, expression, FSM, and toggle—are collected for formal sign-off.

#### **The QuestaSim Command-Line Flow**

A reproducible command-line flow is essential for large-scale regressions in a Linux environment. The following steps demonstrate a typical Questa workflow for coverage collection:

1. **Preparation and Compilation:**  
   Bash  
   \# Create working library  
   vlib work  
   \# Compile RTL and Testbench with line debug and coverage enabled  
   vlog \-sv \-work work \-linedebug \<RTL\_files.v\> \<TB\_files.sv\>

2. **Elaboration and Optimization:**  
   Bash  
   \# Elaborate and enable all coverage types.  
   \# The \+cover=bcesxfT flag is a crucial part of the process.  
   vopt \-coverage \-fsm \-toggle \<top\_module\_name\> \+cover=bcesxfT \-o \<optimized\_design\>

   This step is where the specific coverage metrics are enabled. The \+cover=bcesxfT flag enables branch (b), condition (c), expression (e), FSM (f), statement (s), and toggle (T) coverage.23  
3. **Simulation and Coverage Data Save:**  
   Bash  
   \# Run the simulation, save the coverage data on exit, and quit.  
   vsim \-c \-do "run \-all; coverage save \-onexit \<test\_ucdb\>.ucdb; quit \-f" \<optimized\_design\>

   This command runs the simulation in command-line mode (-c), executes a Tcl script (-do) that runs the test and saves the coverage database (.ucdb) on exit.23

#### **Coverage Database Management: The Power of UCDB**

The Unified Coverage Database (UCDB) is a central component of the Questa verification platform.25 It serves as a unified repository for all coverage metrics, including those collected from C-level, RTL simulation, and formal tools.5

* **Merging UCDBs:** A single test case is never enough to achieve closure. Therefore, a core part of the methodology is merging coverage data from all test runs in a regression.  
  Bash  
  \# Merge all individual UCDBs into a single regression database  
  vcover merge \-out regression.ucdb \-inputs test1.ucdb test2.ucdb test3.ucdb...

  This command combines the coverage results from all individual test runs into a single, comprehensive database, which is used for analysis and final reporting.23  
* **UCIS Export:** For interoperability with other tools or for long-term archival, the UCDB can be exported to the Accellera Unified Coverage Interoperability Standard (UCIS) format.27  
  Bash  
  vcover export \-format ucis \-ucis\_file regression.ucis regression.ucdb

#### **Gate-Level Simulation (GLS) Coverage**

Gate-Level Simulation (GLS) is performed late in the design cycle after the RTL has been synthesized to a gate-level netlist and back-annotated with timing information from an .sdf file.29 GLS is notoriously slow—orders of magnitude slower than RTL simulation.30 Consequently, the goal of GLS is not to repeat the full functional verification but to catch bugs that can only be found in a physical context, such as timing issues, X-propagation, and glitches.

For this reason, code coverage at the gate-level is typically limited to critical metrics:

* **Toggle Coverage:** This is the most valuable metric at the gate level. A full chip-level GLS run can be used to collect toggle coverage to ensure that all internal nodes and signals are transitioning as expected.6 This helps to identify power-hungry sections of the design or to confirm that physical-level optimizations (like clock gating) are not breaking functionality.  
* **FSM Coverage:** A small set of targeted GLS tests can be used to confirm that the FSMs in the design are still fully traversable with the real gate-level timing.32

An important step in this phase is to correlate any remaining coverage holes between the RTL and GLS runs. A hole that was present in the RTL simulation but disappears at the gate level could point to a design issue that was not apparent in the abstract RTL model.

### **6\. Coverage Closure and Sign-off Methodology**

#### **The Requirements-Driven Sign-off Plan**

A structured verification plan is the cornerstone of any successful project. A requirements traceability matrix (RTM) is an ideal tool for this. The RTM links high-level requirements to low-level implementation and verification artifacts, providing a clear path from specification to final sign-off.33

An RTM can be a simple spreadsheet that includes columns for Requirement ID, Description, C++ Testcase ID, RTL Testcase ID, and final coverage metrics like Line Coverage % and Branch Coverage %. This structured approach ensures that every requirement is tied to a specific verification effort and a measurable metric, eliminating guesswork from the sign-off process.

**Table 1: Conceptual Requirements Traceability Matrix**

| Req ID | Req Description | C++ Testcase ID(s) | RTL Testcase ID(s) | Line Cov % | Branch Cov % | FSM Cov % | Sign-off Status |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| IP-01 | FIR filter core functionality | tb\_fir\_fpu | rtl\_regression\_1 | 100% | 98% | N/A | Complete |
| IP-02 | FIR saturation logic | tb\_fir\_fixed\_sat | rtl\_regression\_2 | 100% | 100% | N/A | Complete |
| IP-03 | AXI4-Lite reg access | tb\_axi\_regs\_rw | rtl\_axi\_base | 95% | 90% | N/A | In Progress |
| IP-04 | AXI4-Lite timer IRQ transitions | tb\_axi\_timer | rtl\_axi\_timer\_fsm | 98% | 100% | 100% | Complete |

#### **The Unreachability Problem: Solving the "Last Mile"**

Even with the most exhaustive testbenches, it is nearly impossible to reach 100% code coverage through simulation alone. Some code may be unreachable due to tool limitations, specific design configurations that are not intended to be used, or dead logic that will be optimized away during synthesis. This is often referred to as the "last mile" problem of code coverage.36

The modern solution for this is to use formal verification tools to prove unreachability. Questa CoverCheck and Questa Increase Coverage are automated formal solutions that can mathematically prove that a specific coverage item (a line, a branch, or an FSM state) is unreachable given the design's constraints.36 This is a far more reliable and efficient approach than manual code review, which is time-consuming and prone to human error.39 These tools can automatically generate an exclusion file with a formal proof or rationale for each waived item.39

The final step in coverage closure is to create a formal waiver ledger that documents all excluded coverage items. The ledger should include the location of the coverage item, the reason for the exclusion (e.g., "formally proven unreachable"), and the person responsible for the waiver. This creates an auditable record that demonstrates that all verification gaps have been intentionally closed and justified.

### **7\. Practical Examples and Implementation**

#### **A) Fixed-Point FIR Filter**

The Finite Impulse Response (FIR) filter is a classic Digital Signal Processing (DSP) example that is well-suited for HLS. The filter's C++ implementation uses a loop to perform the weighted sum of input values and includes a fixed-point data type (ac\_fixed) for hardware efficiency.42 The filter's behavior, particularly its saturation logic, provides an ideal target for branch and condition coverage analysis.

* **C-Level Coverage:** The llvm-cov report would highlight which sections of the filter's C++ code were exercised by the C++ testbench. The report would show which iterations of the main processing loop were hit and whether the saturation logic was triggered, indicating that test vectors with large values were applied.  
* **Catapult-Generated RTL and Questa Coverage:** Catapult HLS translates the C++ loop into a pipelined or unrolled datapath with a control FSM.43 A Questa coverage report on the generated RTL would show that the branch coverage on the C++  
  if-else for saturation corresponds to a branch in the generated RTL's FSM. The report would also track FSM state and transition coverage to prove that the control logic for the loop is fully exercised.

#### **B) AXI4-Lite Peripheral (Register File \+ Timer/IRQ)**

An AXI4-Lite peripheral, such as a register file with a simple timer, is a perfect example for demonstrating control flow coverage. The C++ model for this peripheral can be written using Catapult's Matchlib library to describe the AXI4-Lite interface behavior.11 The timer logic would contain a clear state machine for counting, and the register access logic would contain

if-else or case statements for decoding addresses and handling read/write operations.44

* **UVM-lite Testbench:** The RTL testbench for this design would be a "UVM-lite" environment written in SystemVerilog. A simple AXI agent would be responsible for driving transactions to the DUT. The testbench's scoreboard would use DPI-C to call a C++ golden reference model. For each AXI-Lite transaction sent to the DUT, the same transaction is sent to the C++ model, and the results are compared.17  
* **Assertion and FSM Coverage:** A Questa coverage report would be generated to check key metrics.  
  * **FSM Coverage:** The timer's state machine would be explicitly covered, verifying that all states and transitions (e.g., from an IDLE state to a COUNT state, and back to IDLE on an IRQ event) were hit.  
  * **Assertion Coverage:** cover property assertions could be used to monitor AXI-Lite handshakes and temporal behavior, providing a fine-grained view of protocol-level coverage that is independent of the C++ model.

### **8\. Automation, CI/CD, and Reproducibility**

Reproducibility and automation are paramount in a modern ASIC design flow. All steps must be scripted to ensure consistency and to enable seamless integration into a Continuous Integration/Continuous Deployment (CI/CD) pipeline.

#### **Project Skeleton**

A well-structured project repository is key to a reproducible flow:

/my\_hls\_project  
├── hls\_src/  
│   ├── my\_design.cpp  
│   ├── my\_tb.cpp  
│   ├── build.tcl  
│   └──...  
├── rtl\_sim/  
│   ├── rtl/  
│   │   └── Catapult-generated RTL files  
│   ├── tb/  
│   │   ├── my\_tb.sv  
│   │   ├── my\_pkg.sv  
│   │   └──...  
│   ├── run\_dir/  
│   └── Questa/  
│       ├── compile.f  
│       └── sim.do  
├── coverage\_db/  
│   ├── ucdb/  
│   │   ├── per\_test\_ucdbs/  
│   │   └── regression.ucdb  
│   └── html\_reports/  
├── waivers/  
│   └── waiver.xml  
├── Makefile  
└──.gitlab-ci.yml

#### **The Master Makefile**

The Makefile serves as the primary automation script, orchestrating the entire end-to-end flow.

Makefile

**.PHONY**: all clean c\_sim hls rtl\_sim coverage\_merge report

export CATAPULT\_HOME \= /path/to/catapult  
export MTI\_HOME \= /path/to/questa  
export PATH := $(CATAPULT\_HOME)/bin:$(MTI\_HOME)/bin:$(PATH)

all: report

clean:  
    rm \-rf rtl\_sim/rtl/\* rtl\_sim/run\_dir/\* coverage\_db/ucdb/\* coverage\_db/html\_reports/\*

c\_sim:  
    clang++ \-g \-O0 \-fprofile-instr-generate \-fcoverage-mapping hls\_src/my\_design.cpp hls\_src/my\_tb.cpp \-o hls\_src/my\_hls\_test  
   ./hls\_src/my\_hls\_test  
    llvm-profdata merge \-o coverage\_db/ucdb/c\_level.profdata hls\_src/\*.profraw  
    llvm-cov show hls\_src/my\_hls\_test \-instr-profile=coverage\_db/ucdb/c\_level.profdata \> coverage\_db/html\_reports/c\_level.txt

hls:  
    catapult \-f hls\_src/build.tcl \-d rtl\_sim/rtl \-log hls\_src/catapult.log

rtl\_sim:  
    \# Run a Questa regression and collect individual UCDBs  
    \#... commands to run multiple tests and save to ucdb/per\_test\_ucdbs/...

coverage\_merge:  
    vcover merge \-out coverage\_db/ucdb/regression.ucdb \-inputs coverage\_db/ucdb/per\_test\_ucdbs/\*.ucdb

report: coverage\_merge  
    vcover report \-html \-htmldir coverage\_db/html\_reports/final\_report coverage\_db/ucdb/regression.ucdb

This Makefile provides a clear and reproducible flow for local development and can be easily adapted for a CI/CD environment.

#### **CI/CD Pipeline Recipe**

A CI/CD pipeline automates the entire verification flow on every code commit, providing continuous feedback on design quality. A GitLab CI or Jenkins pipeline can be configured to manage this process, sharding regression tests across multiple runners to reduce execution time.46

A sample .gitlab-ci.yml pipeline would include the following stages:

1. **build\_hls:** Runs the Catapult HLS synthesis (make hls) and archives the generated RTL as an artifact.  
2. **rtl\_tests:** Runs the make rtl\_sim target, with tests sharded across multiple parallel jobs. Each job saves its unique UCDB as a job artifact.  
3. **collect\_coverage:** This stage depends on all rtl\_tests jobs. It downloads all UCDB artifacts, merges them (make coverage\_merge), and saves the final regression.ucdb as a pipeline artifact.  
4. **report:** This stage uses the merged UCDB to generate an HTML report (make report) and publishes it as a Web-accessible pipeline artifact.  
5. **sign\_off:** A final stage that uses a scripted check on the regression.ucdb to verify that all coverage goals have been met. If the total coverage is below the sign-off threshold (e.g., 98%), the pipeline fails, providing immediate feedback to the development team.48

This automated approach ensures that the design is continuously verified, that all verification artifacts are securely stored and version-controlled, and that the verification sign-off process is transparent and auditable.

#### **Works cited**

1. Code coverage vs Functional Coverage \- OVM \- Verification Academy, accessed August 29, 2025, [https://verificationacademy.com/forums/t/code-coverage-vs-functional-coverage/25449](https://verificationacademy.com/forums/t/code-coverage-vs-functional-coverage/25449)  
2. What do the terms code coverage and functional coverage refer to when it comes to digital design verification \- Electronics Stack Exchange, accessed August 29, 2025, [https://electronics.stackexchange.com/questions/154580/what-do-the-terms-code-coverage-and-functional-coverage-refer-to-when-it-comes-t](https://electronics.stackexchange.com/questions/154580/what-do-the-terms-code-coverage-and-functional-coverage-refer-to-when-it-comes-t)  
3. Simulation | Siemens Verification Academy, accessed August 29, 2025, [https://verificationacademy.com/topics/simulation/](https://verificationacademy.com/topics/simulation/)  
4. Verification coverage guide \- Tech Design Forum, accessed August 29, 2025, [https://www.techdesignforums.com/practice/guides/verification-coverage/](https://www.techdesignforums.com/practice/guides/verification-coverage/)  
5. Catapult Coverage | Siemens Software \- Siemens EDA, accessed August 29, 2025, [https://eda.sw.siemens.com/en-US/ic/catapult-high-level-synthesis/hls-verification/coverage/](https://eda.sw.siemens.com/en-US/ic/catapult-high-level-synthesis/hls-verification/coverage/)  
6. Using Toggle Coverage \- Application Notes \- Documentation \- Resources \- Support \- Aldec, accessed August 29, 2025, [https://www.aldec.com/en/support/resources/documentation/articles/1511](https://www.aldec.com/en/support/resources/documentation/articles/1511)  
7. SystemVerilog Assertions Part-XXI \- ASIC World, accessed August 29, 2025, [https://www.asic-world.com/systemverilog/assertions21.html](https://www.asic-world.com/systemverilog/assertions21.html)  
8. SVA Quick Reference \- GitHub Pages, accessed August 29, 2025, [https://uobdv.github.io/Design-Verification/Quick-References/SVA\_QuickReference.CDNS.pdf](https://uobdv.github.io/Design-Verification/Quick-References/SVA_QuickReference.CDNS.pdf)  
9. (PDF) Using SystemVerilog Assertions for Functional Coverage \- ResearchGate, accessed August 29, 2025, [https://www.researchgate.net/publication/237566426\_Using\_SystemVerilog\_Assertions\_for\_Functional\_Coverage](https://www.researchgate.net/publication/237566426_Using_SystemVerilog_Assertions_for_Functional_Coverage)  
10. Assert Property vs Cover Property \- SystemVerilog \- Verification Academy, accessed August 29, 2025, [https://verificationacademy.com/forums/t/assert-property-vs-cover-property/30475](https://verificationacademy.com/forums/t/assert-property-vs-cover-property/30475)  
11. Catapult High-Level Synthesis and Verification \- Saros Technology, accessed August 29, 2025, [https://saros.co.uk/wp-content/uploads/2024/02/Catapult-HLS-HLV.pdf](https://saros.co.uk/wp-content/uploads/2024/02/Catapult-HLS-HLV.pdf)  
12. Catapult® High-Level Synthesis \- Amazon S3, accessed August 29, 2025, [https://s3.amazonaws.com/s3.mentor.com/public\_documents/datasheet/hls-lp/catapult-high-level-synthesis.pdf](https://s3.amazonaws.com/s3.mentor.com/public_documents/datasheet/hls-lp/catapult-high-level-synthesis.pdf)  
13. cpp/docs/coverage.md at master · mapbox/cpp \- GitHub, accessed August 29, 2025, [https://github.com/mapbox/cpp/blob/master/docs/coverage.md](https://github.com/mapbox/cpp/blob/master/docs/coverage.md)  
14. llvm-cov \- emit coverage information — LLVM 22.0.0git documentation, accessed August 29, 2025, [https://llvm.org/docs/CommandGuide/llvm-cov.html](https://llvm.org/docs/CommandGuide/llvm-cov.html)  
15. Catapult Basic Training 2010a, accessed August 29, 2025, [https://archive.alvb.in/bsc/TCC/CatapultTutorial/Lab2/Lab2.doc](https://archive.alvb.in/bsc/TCC/CatapultTutorial/Lab2/Lab2.doc)  
16. Catapult High-Level Synthesis and Verification \- Siemens, accessed August 29, 2025, [https://static.sw.cdn.siemens.com/siemens-disw-assets/public/2viQ3qHCWJQxzqSkwBauMQ/en-US/Siemens-SW-Catapult-HLS-HLV-Platform-FS-82981-D1.pdf](https://static.sw.cdn.siemens.com/siemens-disw-assets/public/2viQ3qHCWJQxzqSkwBauMQ/en-US/Siemens-SW-Catapult-HLS-HLV-Platform-FS-82981-D1.pdf)  
17. SystemVerilog for Verification, accessed August 29, 2025, [https://iccircle.com/static/upload/img20240201184635.pdf](https://iccircle.com/static/upload/img20240201184635.pdf)  
18. SystemVerilog for verification. A guide to learning the testbench language features. 2nd revised and expanded ed \- ResearchGate, accessed August 29, 2025, [https://www.researchgate.net/publication/268160090\_SystemVerilog\_for\_verification\_A\_guide\_to\_learning\_the\_testbench\_language\_features\_2nd\_revised\_and\_expanded\_ed](https://www.researchgate.net/publication/268160090_SystemVerilog_for_verification_A_guide_to_learning_the_testbench_language_features_2nd_revised_and_expanded_ed)  
19. How to Call C-functions from SystemVerilog Using DPI-C \- AMIQ Consulting, accessed August 29, 2025, [https://www.consulting.amiq.com/2019/01/30/how-to-call-c-functions-from-systemverilog-using-dpi-c/](https://www.consulting.amiq.com/2019/01/30/how-to-call-c-functions-from-systemverilog-using-dpi-c/)  
20. SystemVerilog DPI-C example (VCS) \- EDA Playground, accessed August 29, 2025, [https://www.edaplayground.com/x/GKL](https://www.edaplayground.com/x/GKL)  
21. Catapult High-Level Synthesis & Verification | Siemens Software, accessed August 29, 2025, [https://eda.sw.siemens.com/en-US/ic/catapult-high-level-synthesis/](https://eda.sw.siemens.com/en-US/ic/catapult-high-level-synthesis/)  
22. Catapult Formal Verification Tools \- Siemens EDA, accessed August 29, 2025, [https://eda.sw.siemens.com/en-US/ic/catapult-high-level-synthesis/hls-verification/formal-verification-tools/](https://eda.sw.siemens.com/en-US/ic/catapult-high-level-synthesis/hls-verification/formal-verification-tools/)  
23. Code coverage Commands with VCS , Questa and IRUN \- forever begin learn, accessed August 29, 2025, [https://svuvm.wordpress.com/2017/03/06/code-coverage/](https://svuvm.wordpress.com/2017/03/06/code-coverage/)  
24. How to generate coverage in VCS (SNPS) , QUESTA and NC, accessed August 29, 2025, [https://www.semiconvn.com/home/tuyen-dung/11208-how-to-generate-coverage-in-vcs-snps--questa-and-nc.html](https://www.semiconvn.com/home/tuyen-dung/11208-how-to-generate-coverage-in-vcs-snps--questa-and-nc.html)  
25. Questa One unified coverage | Siemens Software, accessed August 29, 2025, [https://eda.sw.siemens.com/en-US/ic/questa-one/unified-coverage/](https://eda.sw.siemens.com/en-US/ic/questa-one/unified-coverage/)  
26. Verification Planner in QuestaSim \- Design And Reuse, accessed August 29, 2025, [https://www.design-reuse.com/article/60963-verification-planner-in-questasim/](https://www.design-reuse.com/article/60963-verification-planner-in-questasim/)  
27. Download UCIS (Unified Coverage Interoperability Standard) \- Accellera Systems Initiative, accessed August 29, 2025, [https://www.accellera.org/downloads/standards/ucis](https://www.accellera.org/downloads/standards/ucis)  
28. edaa-org/pyEDAA.UCIS: Unified Coverage Interoperability Standard (UCIS) \- GitHub, accessed August 29, 2025, [https://github.com/edaa-org/pyEDAA.UCIS](https://github.com/edaa-org/pyEDAA.UCIS)  
29. Gate-Level Simulation Methodology \- Multimedia Documents, accessed August 29, 2025, [http://www.multimediadocs.com/assets/cadence\_emea/documents/gatelevel\_simulation\_methodology.pdf](http://www.multimediadocs.com/assets/cadence_emea/documents/gatelevel_simulation_methodology.pdf)  
30. Dan Joyce's 29 tips for gate-level simulation \- DeepChip, accessed August 29, 2025, [https://www.deepchip.com/items/0569-02.html](https://www.deepchip.com/items/0569-02.html)  
31. Gate-Level Simulation Methodology \- Semiconductor Engineering, accessed August 29, 2025, [https://semiengineering.com/gate-level-simulation-methodology/](https://semiengineering.com/gate-level-simulation-methodology/)  
32. QuestaSim \- Sintecs, accessed August 29, 2025, [https://sintecs.eu/webdata/uploads/2024/08/Siemens-SW-QuestaSim-FS-85329-D5.pdf](https://sintecs.eu/webdata/uploads/2024/08/Siemens-SW-QuestaSim-FS-85329-D5.pdf)  
33. How to Create and Use a Requirements Traceability Matrix \- Jama Software, accessed August 29, 2025, [https://www.jamasoftware.com/requirements-management-guide/requirements-traceability/how-to-create-and-use-a-requirements-traceability-matrix/](https://www.jamasoftware.com/requirements-management-guide/requirements-traceability/how-to-create-and-use-a-requirements-traceability-matrix/)  
34. Requirements Traceability Matrix — Everything You Need to Know | Perforce Software, accessed August 29, 2025, [https://www.perforce.com/resources/alm/requirements-traceability-matrix](https://www.perforce.com/resources/alm/requirements-traceability-matrix)  
35. Requirements Traceability Matrix (RTM): A How-To Guide \- TestRail, accessed August 29, 2025, [https://www.testrail.com/blog/requirements-traceability-matrix/](https://www.testrail.com/blog/requirements-traceability-matrix/)  
36. Questa Increase Coverage \- Siemens EDA, accessed August 29, 2025, [https://eda.sw.siemens.com/en-US/ic/questa-one/formal-verification/increase-coverage/](https://eda.sw.siemens.com/en-US/ic/questa-one/formal-verification/increase-coverage/)  
37. Questa CoverCheck Demo \- Siemens Digital Industries Software, accessed August 29, 2025, [https://resources.sw.siemens.com/en-US/product-demo-questa-covercheck-demo/](https://resources.sw.siemens.com/en-US/product-demo-questa-covercheck-demo/)  
38. Questa Formal Verification Apps \- PROLIM, accessed August 29, 2025, [https://www.prolim.com/eda/ic/questa-formal-verification-apps/](https://www.prolim.com/eda/ic/questa-formal-verification-apps/)  
39. Questa Increase Coverage – A Formal etiquette solution for accelerating code coverage closure \- SIEMENS Community, accessed August 29, 2025, [https://community.sw.siemens.com/s/article/Questa-Increase-Coverage-A-Formal-etiquette-solution-for-accelerating-code-coverage-closure](https://community.sw.siemens.com/s/article/Questa-Increase-Coverage-A-Formal-etiquette-solution-for-accelerating-code-coverage-closure)  
40. Questa Covercheck: An Automated Code Coverage Closure Solution, accessed August 29, 2025, [https://semiengineering.com/questa-covercheck-automated-code-coverage-closure-solution/](https://semiengineering.com/questa-covercheck-automated-code-coverage-closure-solution/)  
41. Static Checking for Correctness of Functional Coverage Models | DVCon Proceedings, accessed August 29, 2025, [https://dvcon-proceedings.org/wp-content/uploads/static-checking-for-correctness-of-functional-coverage-models-paper.pdf](https://dvcon-proceedings.org/wp-content/uploads/static-checking-for-correctness-of-functional-coverage-models-paper.pdf)  
42. 2: Catapult design methodology \[16\]. | Download Scientific Diagram \- ResearchGate, accessed August 29, 2025, [https://www.researchgate.net/figure/Catapult-design-methodology-16\_fig14\_291997799](https://www.researchgate.net/figure/Catapult-design-methodology-16_fig14_291997799)  
43. High-level synthesis of digital signal processing circuits \- ERK, accessed August 29, 2025, [https://erk.fe.uni-lj.si/2023/papers/trost(high\_level\_synthesis).pdf](https://erk.fe.uni-lj.si/2023/papers/trost\(high_level_synthesis\).pdf)  
44. Create a custom AXI4 Peripheral \- Arm Learning Paths, accessed August 29, 2025, [https://learn.arm.com/learning-paths/embedded-and-microcontrollers/advanced\_soc/creating\_peripheral/](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/advanced_soc/creating_peripheral/)  
45. hellgate202/axi4\_lib: AXI4 Interface Library \- GitHub, accessed August 29, 2025, [https://github.com/hellgate202/axi4\_lib](https://github.com/hellgate202/axi4_lib)  
46. How GitLab Integration with Jenkins Supercharges Your CI/CD Pipeline \- ONES.com, accessed August 29, 2025, [https://ones.com/blog/knowledge/gitlab-integration-jenkins-supercharge-cicd-pipeline/](https://ones.com/blog/knowledge/gitlab-integration-jenkins-supercharge-cicd-pipeline/)  
47. How To Write CI/CD Pipeline Using GitLab? \- GeeksforGeeks, accessed August 29, 2025, [https://www.geeksforgeeks.org/devops/how-to-write-ci-cd-pipeline-using-gitlab/](https://www.geeksforgeeks.org/devops/how-to-write-ci-cd-pipeline-using-gitlab/)  
48. Guide to Automating API Regression Testing | Sahi Pro Blog, accessed August 29, 2025, [https://www.sahipro.com/post/automating-api-regression-testing-guide](https://www.sahipro.com/post/automating-api-regression-testing-guide)
