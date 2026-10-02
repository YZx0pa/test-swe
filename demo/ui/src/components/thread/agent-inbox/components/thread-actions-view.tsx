import { useCallback, useEffect, useMemo, useState } from "react";
import { Interrupt } from "@langchain/langgraph-sdk";
import { Button } from "@/components/ui/button";
import { InboxItemInput } from "./inbox-item-input";
import useInterruptedActions from "../hooks/use-interrupted-actions";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { buildDecisionFromState } from "../utils";
import { Decision, HITLRequest, DecisionType, ActionRequest } from "../types";
import { useStreamContext } from "@/providers/Stream";

interface ThreadActionsViewProps {
  interrupt: Interrupt<HITLRequest>;
  handleShowSidePanel: (showState: boolean, showDescription: boolean) => void;
  showState: boolean;
  showDescription: boolean;
}

function isValidHitlRequest(
  interrupt: Interrupt<HITLRequest>,
): interrupt is Interrupt<HITLRequest> & { value: HITLRequest } {
  return (
    !!interrupt.value &&
    Array.isArray(interrupt.value.action_requests) &&
    interrupt.value.action_requests.length > 0 &&
    Array.isArray(interrupt.value.review_configs) &&
    interrupt.value.review_configs.length > 0
  );
}

function getDecisionStatus(
  decision: Decision | undefined,
): DecisionType | null {
  if (!decision) return null;
  return decision.type;
}

function getActionTitle(action?: ActionRequest) {
  return action?.name ?? "Unknown interrupt";
}

// Jeni demo: the card shows the server's plain-language summary of the call, not its payload,
// the thread's state, its id or Studio.
export function ThreadActionsView({ interrupt }: ThreadActionsViewProps) {
  const stream = useStreamContext();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [addressedActions, setAddressedActions] = useState<
    Map<number, Decision>
  >(new Map());
  const [submittingAll, setSubmittingAll] = useState(false);

  const hitlValue = interrupt.value;
  const actionRequests = useMemo(
    () => hitlValue?.action_requests ?? [],
    [hitlValue?.action_requests],
  );
  const reviewConfigs = useMemo(
    () => hitlValue?.review_configs ?? [],
    [hitlValue?.review_configs],
  );

  const hasMultipleActions = actionRequests.length > 1;
  const currentAction = actionRequests[currentIndex];
  const matchingConfig =
    reviewConfigs.find(
      (config) => config.action_name === currentAction?.name,
    ) ?? reviewConfigs[currentIndex];

  const singleActionInterrupt = useMemo(() => {
    if (!currentAction || !matchingConfig) {
      return interrupt;
    }

    return {
      ...interrupt,
      value: {
        action_requests: [currentAction],
        review_configs: [matchingConfig],
      },
    };
  }, [interrupt, currentAction, matchingConfig]);

  const {
    approveAllowed,
    hasEdited,
    hasAddedResponse,
    streaming,
    supportsMultipleMethods,
    loading,
    handleSubmit,
    setSelectedSubmitType,
    setHasAddedResponse,
    setHasEdited,
    humanResponse,
    setHumanResponse,
    selectedSubmitType,
    initialHumanInterruptEditValue,
  } = useInterruptedActions({
    interrupt: singleActionInterrupt,
  });

  useEffect(() => {
    setCurrentIndex(0);
    setAddressedActions(new Map());
  }, [interrupt]);

  const handleApproveAll = useCallback(() => {
    if (!hasMultipleActions) return;

    try {
      const allDecisions: Decision[] = actionRequests.map(() => ({
        type: "approve",
      }));

      stream.submit(
        {},
        {
          command: {
            resume: { decisions: allDecisions },
          },
        },
      );

      toast("Success", {
        description: "All actions approved successfully.",
        duration: 5000,
      });
    } catch (error) {
      console.error("Error approving all actions", error);
      toast.error("Error", {
        description: "Failed to approve all actions.",
        richColors: true,
        closeButton: true,
        duration: 5000,
      });
    }
  }, [actionRequests, hasMultipleActions, stream]);

  const handleSubmitAll = useCallback(() => {
    if (!hasMultipleActions) return;

    if (addressedActions.size !== actionRequests.length) {
      toast.error("Error", {
        description: `Please address all ${actionRequests.length} actions before submitting.`,
        richColors: true,
        closeButton: true,
        duration: 5000,
      });
      return;
    }

    try {
      setSubmittingAll(true);
      const allDecisions = actionRequests.map((_, index) => {
        const decision = addressedActions.get(index);
        if (!decision) {
          throw new Error(`Missing decision for action ${index + 1}`);
        }
        return decision;
      });

      stream.submit(
        {},
        {
          command: {
            resume: { decisions: allDecisions },
          },
        },
      );

      toast("Success", {
        description: "All actions submitted successfully.",
        duration: 5000,
      });
      setAddressedActions(new Map());
    } catch (error) {
      console.error("Error submitting all actions", error);
      toast.error("Error", {
        description: "Failed to submit actions.",
        richColors: true,
        closeButton: true,
        duration: 5000,
      });
    } finally {
      setSubmittingAll(false);
    }
  }, [actionRequests, addressedActions, hasMultipleActions, stream]);

  const allAllowApprove = useMemo(() => {
    if (!hasMultipleActions) return false;
    return actionRequests.every((actionRequest) => {
      const matching = reviewConfigs.find(
        (config) => config.action_name === actionRequest.name,
      );
      return matching?.allowed_decisions.includes("approve");
    });
  }, [actionRequests, reviewConfigs, hasMultipleActions]);

  const handleSaveDecision = () => {
    const { decision, error } = buildDecisionFromState(
      humanResponse,
      selectedSubmitType,
    );

    if (!decision || error) {
      toast.error("Error", {
        description: error ?? "Unable to determine decision.",
        richColors: true,
        closeButton: true,
        duration: 5000,
      });
      return;
    }

    setAddressedActions((prev) => {
      const next = new Map(prev);
      next.set(currentIndex, decision);
      return next;
    });

    toast("Success", {
      description: `Action ${currentIndex + 1} captured.`,
      duration: 3000,
    });

    if (currentIndex < actionRequests.length - 1) {
      setCurrentIndex((prev) => Math.min(actionRequests.length - 1, prev + 1));
    }
  };

  const currentTitle =
    currentAction?.description || getActionTitle(currentAction);
  const actionsDisabled = loading || streaming || submittingAll;
  const hasAllDecisions =
    hasMultipleActions && addressedActions.size === actionRequests.length;

  if (!isValidHitlRequest(interrupt)) {
    return (
      <div className="flex min-h-full w-full flex-col items-center justify-center rounded-2xl bg-gray-50/50 p-8">
        <p className="text-sm text-gray-600">
          Unable to render interrupt. The data provided is not in the expected
          HITL format.
        </p>
      </div>
    );
  }
  const interruptValue = singleActionInterrupt.value as HITLRequest;

  return (
    <div className="flex min-h-full w-full max-w-full flex-col gap-9">
      <div className="flex w-full flex-col gap-1">
        <p className="text-sm font-medium text-amber-700">
          Needs your approval
          {hasMultipleActions &&
            ` (${currentIndex + 1} of ${actionRequests.length})`}
        </p>
        <p className="text-xl tracking-tight text-pretty">{currentTitle}</p>
      </div>

      <div className="flex w-full flex-row flex-wrap items-center justify-start gap-2">
        {hasMultipleActions && allAllowApprove && (
          <Button
            variant="outline"
            className="border-gray-500 bg-white font-normal text-gray-800"
            onClick={handleApproveAll}
            disabled={actionsDisabled}
          >
            Approve All
          </Button>
        )}
      </div>

      {hasMultipleActions && (
        <div className="flex w-full items-center gap-2">
          {actionRequests.map((_, index) => {
            const status = getDecisionStatus(addressedActions.get(index));
            return (
              <button
                type="button"
                key={index}
                onClick={() => setCurrentIndex(index)}
                className={cn(
                  "h-2 flex-1 rounded-full border transition-colors",
                  "border-gray-300 bg-gray-200",
                  status === "approve" && "border-emerald-500 bg-emerald-200",
                  status === "reject" && "border-red-500 bg-red-200",
                  status === "edit" && "border-amber-500 bg-amber-200",
                  index === currentIndex &&
                    "outline-primary outline-2 outline-offset-2",
                )}
              >
                <span className="sr-only">Action {index + 1}</span>
              </button>
            );
          })}
        </div>
      )}

      <InboxItemInput
        approveAllowed={approveAllowed}
        hasEdited={hasEdited}
        hasAddedResponse={hasAddedResponse}
        interruptValue={interruptValue}
        humanResponse={humanResponse}
        initialValues={initialHumanInterruptEditValue.current}
        setHumanResponse={setHumanResponse}
        supportsMultipleMethods={supportsMultipleMethods}
        setSelectedSubmitType={setSelectedSubmitType}
        setHasAddedResponse={setHasAddedResponse}
        setHasEdited={setHasEdited}
        handleSubmit={hasMultipleActions ? handleSaveDecision : handleSubmit}
        isLoading={hasMultipleActions ? submittingAll : loading}
        selectedSubmitType={selectedSubmitType}
      />

      {hasMultipleActions && (
        <div className="flex w-full items-center justify-between">
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={currentIndex === 0}
              onClick={() => setCurrentIndex((prev) => Math.max(0, prev - 1))}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={currentIndex === actionRequests.length - 1}
              onClick={() =>
                setCurrentIndex((prev) =>
                  Math.min(actionRequests.length - 1, prev + 1),
                )
              }
            >
              Next
            </Button>
          </div>
          <Button
            variant="brand"
            disabled={!hasAllDecisions || submittingAll}
            onClick={handleSubmitAll}
          >
            {submittingAll
              ? "Submitting..."
              : `Submit all ${actionRequests.length} decisions`}
          </Button>
        </div>
      )}
    </div>
  );
}
