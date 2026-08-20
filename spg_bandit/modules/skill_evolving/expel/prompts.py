"""Prompt assets and builders derived from the bundled ExpeL source.

The original repository is vendored under ``docs/ExpeL``.  We read its
literal ALFWorld examples through the Python AST rather than importing its
legacy LangChain/OpenAI stack, so the embedded runtime uses the exact source
examples without acquiring those old dependencies.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASK_PREFIX = {
    "pick_and_place_simple": "put",
    "pick_and_place": "put",
    "pick_clean_then_place_in_recep": "clean",
    "clean": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "heat": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "cool": "cool",
    "look_at_obj_in_light": "examine",
    "examine": "examine",
    "pick_two_obj_and_place": "puttwo",
}


# Exact ALFWorld assets extracted from LeapLabTHU/ExpeL commit
# e41ec9a24823e7b560c561ab191441b56d9bcefc.  The compressed fallback makes
# the adapter self-contained when the reference checkout under docs/ is not
# copied to a server.  When that checkout is available, load() still audits
# the live source and records its hash.
_OFFICIAL_SOURCE_SHA256 = "130356ae2f6e08b9c90447bbf552ea279b4895762e87fe83be4347aff3f0d043"
_EMBEDDED_ASSETS_B85 = """c-rk<-H+Qg68~3(eM$j$FP3(lCJzPjQe1FF?xDGd;t~{Gn_demi#|z?vncw%-|$PEA*nB~y}N1hkVKRza!Ah1Z#bM`{=BH~>xOUt$jiE^_RA(0<;9zeUqn%eL(*(`Qs0*i`;v&YD)BFDyDNB|EDKTd;yyWSc$qZpE-!BrrarGPFJ>1Nhr%DW_Qjh&FO~&o(DZ*Vei!?MRXoW{)#5f^tqLyNXGtZ*_A>cZ2-V#G_WAM-x?q*qm#gGn>}T??q~;vzF4;|9@+O&QGh9kZX--Nnap^TK{DcdyaAAQ9*SK(n3xo(tgDQyin)5ANZs47LwJ#f9HDagBtGt9kH|(Y`6;>5H@Jc_gt9*59-rut7hTZZdD|LCBFDr3ipUuzph8Js1kA6(uqppybcQ*|FHpNDK<~RGQgubr)7n%Pe^<T&~U~yQ|2nLJ<FD^fpu*4NGMSYj#HT>`<+3g!SWXS?a!JB5kRO^q%3}5{W`~bAQNj`|A76lAQ@`r3@mb^<!&R5V@vd(39@CTSq@_}h?CM%bwi@Zq=d9!gtHNQ$S;5V=5<YW2of8qXr$*Z~vy0krACU4ys>SQZ`my7&P3Io(-yB%kh?82<si}?>TvbgGnTLfK1Gx}owV+OA*|3Gn`{cs87$Q561`0_3hz(hCq@*nvrs~RaTYQ&aNF55BP3N-}t{$u$8=G2xX{YsEms;b!W64jJ~s$Mndlgq?!Kvn_r<g63<-0w8|@Y`F>HapQUASJULW6CKalL!F}LHF&6hU)_++e*s~Xq=UK+&938cV)ijMihY*^{^Jzoqj^U>PPiu!B)0CyTV$;_GSq3LA}|*TIfM>Sn#Ue05cj{t#_>17fcVn5nDAzCnc&p$L#hVH0?W&$s}7`Chz%|2J^mR`Mhz9t6sFE!m^01KZkM^cc4fUm;^bQUhQChL&~V2#VOOvN4Q#z*vg~Nk4Oo%%S;rLzsRXuAFYFSLX=qhsCms)7gW#8JzZcYMc1NJvah8idBlV>qCXv&l2+OLLzKZ3lgW5KTz@7;ms8;;IWi-LO`#fH4_!6JqhMp4*pN6Qo@f4x^rVQNuQ+PuDezqjL@l=5aHssnp>0dyu_#~aW+#F??sg2qn_d9m&C>{cuVVnO0NgE$s)6_;W&xlzR@&rigL^BpDN!R9;-gyO2<2A_xq!DFpQq8!Q%vR#<kjfUjz+R*BN=YwMjQ$~KJ|n&aXV0>m<i&wQi>1{NiH>=LL-DI;0(=fE21?k8mpjD&sa0HT~l?1#)e?T=&^bOU|Skxq&pI9cO#m{o5l&C-hz{~4(AwguxXf~#<3D?Tc?`Qhsf5_l$~T<CYP1<7ujfw%QUYoC`y=bZjq`!$z8TqFG*)KwWh3!v4PC+hD<D5Z<>MCI}gOZLXs4Y|M3M+?RU+P0|KeBL#uhjNkiw;>|!ZIaSqoBac<0scVkY>p}&EXC7t4Qxkk@d=s82r=|fC5wth-$4cJK=Yda-n#8h*w9qp2VR;{Hbd$NX_o}bW4BR!7t#6SniNR#CEnP{F{<tjA00>=z7(|$}nN6bQgqL&<s*Q4z;vPSJFjef$vISoZuHDUmTU#HQpsr@4w3fY*wqMsAXXXj8$F10FZX!9D1J2!ULx*dR7A4E|<0ePMNiq6jT$Qg3bY5df@dXy=xzSUF`yWGH_{MC)aVIVT!eE^i7^o)Um-x;_|^`PmEbV4BvR|{G{%U=B%;Yv0NXSB|Ejh=r(&#%z);xS?98J`tM0g%?GRO6DMqtOT&nSw^9uTeMzP0P~&MP?5SQA|<JeZo>2cpS{UmN=yr!BlsU^LjMoU>5Xs+N2#J+6y{oAVa>_UBMdOwc|`<?WZ07-N1*bT$hTmHez3M=&3Y9i7HuRJ;P4PmbwGXYGkj9B{)W6*5z>pJ=N=bRaD#-tgcP3ss94epUr&N3Eg?HouuZ#i%Jy6(V%BLqdsuC_RSvc&_A=9lam!siJ>8T%B|bg(kev_g>5nBRzh`&U#Wes9ZRH*sAs90jFgg*_K}fxkdaa{(j&;oC>dEF8CeGz86_j@@w52`TnlJ!MIFQqQ-4jVzwV>{x`X;_O8xZ_lenN{Ec(b;bda&2WGs3nQ7Y_Ce~|ihf-^`prj1jNL~O#MT9oK6^R<(TbD6K2!anm=*BEWSnoov|S8Z(sUK{<|S*+G5iFm78w@q+H)$N_pV<99^qZU^{1Q?HujBJ4gJfTzOh8r+h@%oNfjlI^MZ{Z=&f+G0ZJzkA8`K<S;b!JnCRA8_b5)xUXo#8MtAWgZ?VH^e9?v6IxWP8RwYkvYGz)X>9T8C~V!<L$COPXqm#y(rknOe&5l51g(Yuss$atpKu+V#<=H+CI^^I-GdokX+?-g6LQ7(@%YEyWb6t4Z{zeXZ3fZeXM`x9mzN$a*Nq;uK_rf@}%}N_5<`8A(EhT*0a>P_w+V=Qq}xTk@jVZG_$6-SPY7hS_~j!FKJ|&OAHo7l_Y}bT_Z8+A2wV%e<gYnju{n&V-R|?~^n*#p(`3Q9n<JLeb3`XOQjPM?qW(k<|%;-9#8FohEqOKKIM_NQsu}SbGSOLKrRBitz%m{+n{A1B^(MX~zIts8cIO=uJC$vaKIw35gAuEoIsDpqa7RS7jnf0#GORQ8^$tTKnZ=dw^CIuj@_&L2>zO_xO{0eB~Y&6HN$hD6}?GU?doWN&5?7k(HdB#7uzJhOC6HV+08*ZxuP?OoI7KtR`M#HE}Kr03qSL2TJF0D4i21olghzZ`+(Oaz`k*>Y?B&PQewS;HocZcBSQtDEJYIQU3ZO9gWqhF%vCHC?|(5t`Zqb=XpyP4M#(@QDYjdM@Zd_it2bFH0m;xLh}PM#9PhU4ck7-As9Evu&nqhZ<efDMR3q9ern-2!(1?b5_p6??}oGH98n0*HDVxpNCd&79^*X0DUD<X9YY{vg=h@H3DP=B*{p$e)s;QAX@}mM(lIvP6FO0W1ab^xb2-A(oG`m1j61NC8C6?2#Ng!vf<Mn6(b*N9H!a4{ot&rxo(TbrL&r%G7xR>Z9u&{%$3eDfEO*=;>zJc2D}-ny<`^pNjyjYLPQYx{zi~5@A^EtCD*8k1jyH@GV_nTo-&k+6kx3shkz&$D6MEs<O{xV=D-~?ZxvGL2kaIhun&E)dSADa4=Cq8|9SvqAlG3ov+)KVdi58KwKJ3p<y~slnh3oU3b9C$38_c=D)gC9`;(OxeR}+k}5jLE)#lq8Ma*Txbw###6adZX)%GF0N+Wke4p!xM@aM)9{He!B$>J&8>&EZngS_1$jjfC;m3TCYW476Vm_+cU=P>_U|Gl-)?vPh&zI+!dlpO`E#E;*Cb9i5ulOVrsh+|oRN;<nmE9Pb$!LGtNxG))X#ME~6DF&55`qahJxau*F1&#;M}hG`VQd|cYV*p(r&5yi?vhyT(;q%;ChjmQhzr+!v84zzSj@dvfqg+XekS??-#ZzNPsu4?mWzcDC`%CtrS2RA(`3=q*Jom-g9-mpLiAXkwDK{I_pvY<MvIu2}H^pzhF=)$$UC?a|!RKWHMsb5xwmgrMQYDs1<SJWnU>(T8(#kB^uWk#`+c4+`m<mfX}n~tx`uR2H=(Sug#{1vwE+XZ&g>1GbCW7=vxN;vI>>b_`A)YbR2?Pr#@tyurDvbLW(FS`yBvuod>S9{vF{+^rtJvaM%ZuYa!%|3ft_6Taep>$-Z+Z@CRcQNw3bi;CkV{J(buMMXe>vx+L4pT6or@G&;?T*9@_lLZy**XY(kn0YtkAzF5+7<q}R>MOs-Gv|U(l$`@rI5kkgDs4M-WW}Ci-X5KDsm>(I7=@;A@`#M&bO8g%;)CN|7#1aiJbp?NBPes;ARpQxroHni)gs=%9&TzR!Ck|qG|&SWlUijjWG<>aEvGWcWj1Z+OGp4ks3XbK1M%XZ|?c>@Gye2xCqRI_5C^R*!GPdBt6f1U?>X#`i2+$b6)47L@+`|<aJ7sBvO;~8;;z584Mh$t$1`%bXD!Tukwm78_-Q5n?EQqa-s-4kmaLHJJtt{X_?5k*<`*QN$E)$H_$Ir7&#lw$XN#?pU|y=j+}bUoD)R@lWZT)Ct>yzYc^m6wQ?Z$_E7!d29RwrbR9tRgT_zq*OLt&+nTa_+)#V-z5Qdu@cLIT_rO_7>XH&U=jOl>#XuhE^u-(%Dm?U~mlPb?k=%$cenLVPO9R(R$JMbqhjbPw9lw4+%CJSyHZfw4W*M7q&(vGDGgC!81tmY!>Kwy(QF3)twwjcRZ$%?{u&D%6?({X%Kxn*}Bv1PUwH#f85E@A4Fnl_8f0t~evr=l#hOa`du0{02{vE-ym4t~W+#dMTm|c{GV>9zzj)ck79|84_na78TA-YDLI0yB|o;r^w^b}G@_03s?FdZd?X-o(&KK&)Hyk8ew-GGnoz5jNir3a&<dk;^(qEP7GU}$M~(o+;NYbUtGhJ;KZVgq9*>{uphYrcLkLt^j28ScnQ!z*=DFBfx0Lbv+w)As_fkGw+XY!i5jqkz^|+=f=QcUjE)Z@QR|ajmO%#_SwX#gSXa9x}RkU9V}!HFkryoSA!!)aN!W@!M_Sw-0YW{O30i-?#66{aC)|2WiNcZ(tP;tcK6`u(r9n>|Vek!qZ`sFE{Frj}k;>&kEphL&J)8m&pHAMxM2SY{;dQGNJh+aw#ITSZ~C>SZS+MvPRj(c1RUMpD_lR_p3wIUjr&;B?S>0Dt3~M_!qw1!zM=VYQZjr@hw?l|63b93jj8f)B{)?NmJe9uOzxb9~FFOzDe>m)MeXX8rdz_ikIieIhIM~Y4jSA2WKP=ADslZk32Ao_<3D6a+X_07Vef2dI`ZIk|#uG{J6uAr#lBhMs)PtgI;z*LIYerY&1dmy#`az95BOW{fH^NK~LBIIIhDw3qsAJhnhv4ngyX|F?9BnhDpTJZ={GMs?xB4ei!!W?9_>gsXr&Z%dgZS9bh^XF0wQ}WPDnOeZT`(ri_Xm)cu+rx*;-pSqk2jt9m_Sj=k-KcR3@#H@y6uSEAekXw8!Cz4}!lKh<gj7cBf!9QbFm2hr-fX7{R%A${$^zN`*Y21YDhYSq2>qt>?XB!cHlInNstYTkY&$FWv;DN1`Oiyk=`;IKr1;5jR17SFLz%$z$$HZ^k;kp5#sy7=_h{{cca-*N"""


OPERATION_FORMAT = """<OPERATION> <RULE NUMBER>: <RULE>

The available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. """


@dataclass(frozen=True)
class OfficialPromptAssets:
    benchmark: str
    system_instruction: str
    react_examples: dict[str, list[str]]
    reflection_examples: list[str]
    source_path: str
    source_sha256: str

    @classmethod
    def load(
        cls, source_dir: str | Path | None = None, benchmark: str = "alfworld",
    ) -> "OfficialPromptAssets":
        benchmark = str(benchmark).lower()
        if benchmark not in {"alfworld", "webshop"}:
            raise ValueError(f"ExpeL prompt assets do not support benchmark {benchmark!r}")
        explicitly_configured = source_dir is not None
        if not explicitly_configured:
            source_dir = Path(__file__).resolve().parents[4] / "docs" / "ExpeL"
        source_path = Path(source_dir) / "prompts" / f"{benchmark}.py"
        if not source_path.is_file():
            # The bundled fallback is an audited ALFWorld snapshot only.  Using it
            # for WebShop silently changes the asset benchmark and later fails
            # with a misleading missing-task-type error.
            if explicitly_configured or benchmark != "alfworld":
                raise FileNotFoundError(
                    f"ExpeL {benchmark} prompts were not found at {source_path}. "
                    f"Provide docs/ExpeL/prompts/{benchmark}.py or configure "
                    "skill_evolving.official_source_dir to the ExpeL source tree."
                )
            payload = json.loads(zlib.decompress(
                base64.b85decode(_EMBEDDED_ASSETS_B85.encode("ascii"))
            ))
            return cls(
                benchmark="alfworld",
                system_instruction=str(payload["system_instruction"]),
                react_examples={
                    str(key): [str(item) for item in value]
                    for key, value in payload["react_examples"].items()
                },
                reflection_examples=[str(item) for item in payload["reflection_examples"]],
                source_path=(
                    "embedded:LeapLabTHU/ExpeL@"
                    "e41ec9a24823e7b560c561ab191441b56d9bcefc"
                ),
                source_sha256=_OFFICIAL_SOURCE_SHA256,
            )
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))

        def literal(name: str) -> Any:
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError) as exc:
                        raise ValueError(f"ExpeL prompt asset {name} is not a literal") from exc
            raise KeyError(f"ExpeL prompt asset {name} was not found in {source_path}")

        if benchmark == "webshop":
            return cls(
                benchmark="webshop",
                system_instruction=str(literal("SYSTEM_INSTRUCTION")),
                react_examples={"webshop": [str(item) for item in literal("FEWSHOTS")]},
                reflection_examples=[str(item) for item in literal("REFLECTION_FEWSHOTS")],
                source_path=str(source_path),
                source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )
        examples = literal("d")
        reflections = literal("REFLECTION_FEWSHOTS")
        instruction = literal("SYSTEM_INSTRUCTION")
        react_examples = {
            prefix: [examples[f"react_{prefix}_{index}"] for index in range(2)]
            for prefix in set(TASK_PREFIX.values())
        }
        source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if source_sha256 != _OFFICIAL_SOURCE_SHA256 and not explicitly_configured:
            raise ValueError(
                "docs/ExpeL prompt source differs from the audited embedded snapshot; "
                "set official_source_dir explicitly to opt into another ExpeL revision"
            )
        return cls(
            benchmark="alfworld",
            system_instruction=str(instruction),
            react_examples=react_examples,
            reflection_examples=[str(item) for item in reflections],
            source_path=str(source_path),
            source_sha256=source_sha256,
        )

    def examples_for(self, task_type: str) -> list[str]:
        if self.benchmark == "webshop":
            return list(self.react_examples["webshop"])
        prefix = TASK_PREFIX.get(task_type)
        if prefix is None:
            raise ValueError(f"No official ExpeL ALFWorld prompt mapping for {task_type!r}")
        return list(self.react_examples[prefix])


def format_task_memory(reflections: list[str]) -> str:
    """The reference implementation's ``PREVIOUS_TRIALS_FORMATTER``."""
    if not reflections:
        return ""
    lines = ["Your memory for the task below:"]
    for index, reflection in enumerate(reflections):
        lines.extend((f"Trial {index}:", reflection.strip()))
    return "\n".join(lines)


def format_alfworld_task(initial_observation: str, task_goal: str) -> str:
    """Recreate the task string used by ExpeL's bundled ALFWorld data."""
    return (
        f"{initial_observation.strip()}\n"
        f"Your task is to: {task_goal.strip()}"
    )


def format_webshop_task(task_goal: str) -> str:
    """Match ExpeL's WebShop task representation."""
    return f"Instruction:\n{task_goal.strip()}\n[Search]"


def build_actor_context(
    *,
    assets: OfficialPromptAssets,
    task_type: str,
    task_goal: str,
    initial_observation: str,
    max_steps: int,
    rules: str,
    retrieved_demonstrations: list[dict[str, Any]],
    reflections: list[str],
) -> tuple[str, str]:
    """Build the immutable part of one ExpeL ReAct trial."""
    official_examples = assets.examples_for(task_type)
    if assets.benchmark == "webshop":
        learned_examples = [
            f"{format_webshop_task(str(item['task_goal']))}\n{str(item['trajectory']).strip()}"
            for item in retrieved_demonstrations[:2]
        ]
        examples = learned_examples + official_examples[:max(0, 2 - len(learned_examples))]
        sections = [
            f"You may take maximum of {max_steps} steps.\nHere are two examples:\n\n"
            + "\n\n".join(examples) + "\n\n(END OF EXAMPLES)"
        ]
        if rules:
            sections.append(
                "The following are experiences gathered while purchasing requested items "
                "from an online store. Use them as useful references:\n" + rules
            )
        task_memory = format_task_memory(reflections)
        if task_memory:
            sections.append(task_memory)
        sections.append("Now it's your turn!\n" + format_webshop_task(task_goal) + "\n\nAction:")
        return assets.system_instruction, "\n\n".join(sections)
    learned_examples = [
        f"{format_alfworld_task(str(item.get('initial_observation', '')), str(item['task_goal']))}"
        f"\n{str(item['trajectory']).strip()}"
        for item in retrieved_demonstrations[:2]
    ]
    # During evaluation the source implementation replaces its two static
    # examples with retrieved trajectories.  Keep the same two-example budget
    # and fill any missing slots with the official demonstrations.
    examples = learned_examples + official_examples[:max(0, 2 - len(learned_examples))]
    sections = [
        f"You may take maximum of {max_steps} steps.\nHere are two examples:\n\n"
        + "\n\n".join(examples)
        + "\n\n(END OF EXAMPLES)",
    ]
    if rules:
        sections.append(
            "The following are some experience you gather on a similar task of "
            "completing a household task by interacting in a household environment. "
            "Use these as references to help you perform this task:\n" + rules
        )
    task_memory = format_task_memory(reflections)
    if task_memory:
        sections.append(task_memory)
    sections.append(
        "Now it's your turn!\n"
        + format_alfworld_task(initial_observation, task_goal)
    )
    return assets.system_instruction, "\n\n".join(sections)


def build_reflection_prompt(
    *, assets: OfficialPromptAssets, initial_observation: str,
    task_goal: str, trajectory: str,
) -> tuple[str, str]:
    """Build ExpeL/Reflexion's task-specific ``New plan`` prompt."""
    if assets.benchmark == "webshop":
        user = (
            "Here are two examples:\n\n" + "\n\n".join(assets.reflection_examples)
            + "\n\n(END OF EXAMPLES)\n\nPrevious Trial:\n"
            + format_webshop_task(task_goal) + f"\n{trajectory.strip()}\n\nNext plan:"
        )
        return "You improve an online-shopping agent after a failed purchase trial.", user
    system = (
        "You are an advanced reasoning agent that improves from self-reflection. "
        "Analyze the failed household-task trial and propose a concrete new plan."
    )
    user = (
        "Here are two examples:\n\n"
        + "\n\n".join(assets.reflection_examples)
        + "\n\n(END OF EXAMPLES)\n\n"
        + "Previous trial:\n"
        + format_alfworld_task(initial_observation, task_goal)
        + f"\n{trajectory.strip()}\n"
        + "STATUS: FAIL\nNew plan:"
    )
    return system, user


def build_insight_prompt(
    *,
    kind: str,
    rules: str,
    success_history: str,
    failure_history: str | None = None,
    task_goal: str | None = None,
    list_full: bool = False,
) -> tuple[str, str]:
    """Build the two insight-extraction prompts used by official ExpeL."""
    existing = rules or "(none)"
    fullness = (
        "Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY "
        "insightful and different from EXISTING RULES. Below are the operations you "
        "do to the above list of EXISTING RULES:"
        if list_full
        else "Below are the operations you do to the above list of EXISTING RULES:"
    )
    if kind == "compare":
        system = (
            "You are an advanced reasoning agent that can add, edit or remove rules from "
            "your existing rule set, based on forming new critiques of past task "
            "trajectories. You will be given two previous task trials in which you were "
            "placed in a household environment and a task to complete: one successful "
            "and one unsuccessful trial. You failed the trial because you reached the "
            "maximum allowed number of steps without completing the task."
        )
        user = f"""Here are the two previous trials to compare and critique:
TRIAL TASK:
{task_goal or ''}

SUCCESSFUL TRIAL:
{success_history}

FAILED TRIAL:
{failure_history or ''}

Here are the EXISTING RULES:
{existing}

By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:

{OPERATION_FORMAT}

{fullness}"""
    elif kind == "all_success":
        system = (
            "You are an advanced reasoning agent that can add, edit or remove rules from "
            "your existing rule set, based on forming new critiques of past task "
            "trajectories. You will be given successful tasks trials in which you were "
            "placed in a household environment and tasks to complete."
        )
        user = f"""Here are the trials:
{success_history}

Here are the EXISTING RULES:
{existing}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:

{OPERATION_FORMAT}

{fullness}"""
    else:
        raise ValueError(f"Unsupported ExpeL insight kind: {kind}")
    return system, user
